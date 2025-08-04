from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig


from ..MoRA.modeling_llama_mora import  LlamaMoRAForCausalLM, LlamaMoRAModel
from llava.constants import IMAGE_TOKEN_INDEX
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..llava_mora_arch import LlavaMoRAMetaModel, LlavaGraphMoRAMetaForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput

class LlavaMoRAConfig(LlamaConfig):
    model_type = "llava_mora"


class LlavaLlamaMoRAModel(LlavaMoRAMetaModel, LlamaMoRAModel):
    config_class = LlavaMoRAConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaMoRAModel, self).__init__(config)


class LlavaLlamaMoRAForCausalLM(LlamaMoRAForCausalLM,LlavaGraphMoRAMetaForCausalLM):
    config_class = LlavaMoRAConfig

    def __init__(self, config):
        super(LlamaMoRAForCausalLM, self).__init__(config)
        self.model = LlavaLlamaMoRAModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_mora_weights(self, images, input_ids):
        if images is None:
            return None
        mora_weights = self.encode_graphs(images)

        # fill zero for language-only data
        mask = input_ids == IMAGE_TOKEN_INDEX
        mask = (mask.sum(1) > 0).long().reshape(-1, 1, 1)
        for mora_weights_sub in mora_weights:
            for key in mora_weights_sub:
                if mora_weights_sub[key] != (None, None):
                    mora_weights_sub[key] = (mora_weights_sub[key][0] * mask, 
                                              mora_weights_sub[key][1])

        return mora_weights

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graphs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        mora_weights = None,
        graph_token_idx = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if mora_weights is None:
            mora_weights = self.get_mora_weights(graphs, input_ids)
        
        assert mora_weights is not None

        # TODO: an ugly way to handle IMAGE_TOKEN_INDEX
        mask = input_ids == IMAGE_TOKEN_INDEX
        input_ids[mask] = 0
        if mask.shape != attention_mask.shape:  # not equal when .generate()
            assert graph_token_idx is not None
            for i in range(graph_token_idx.shape[0]):
                if graph_token_idx[i] >= 0:
                    attention_mask[i, graph_token_idx[i]] = 0
        else:
            attention_mask[mask] = 0
        
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mora_weights=mora_weights
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        graphs = kwargs.pop("graphs", None)
    
        mora_weights = self.get_mora_weights(graphs, inputs)

        # Find graph token indices
        graph_token_idx_list = []
        for i in range(inputs.shape[0]):
            mask = (inputs[i] == IMAGE_TOKEN_INDEX).int()
            if mask.sum() > 0:
                graph_token_idx_list.append(mask.argmax().item())
            else:
                graph_token_idx_list.append(-1)
        graph_token_idx = torch.Tensor(graph_token_idx_list).long().to(inputs.device)

        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            # assistant_model, # fixme: remove for transformers 4.28.0
            streamer,
            mora_weights=mora_weights,
            graph_token_idx=graph_token_idx,
            **kwargs
        )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        graphs = kwargs.pop("graphs", None)
        mora_weights = kwargs.pop("mora_weights", None)
        graph_token_idx = kwargs.pop("graph_token_idx", None)
        
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        
        if graphs is not None:
            _inputs['graphs'] = graphs
        if mora_weights is not None:
            _inputs['mora_weights'] = mora_weights
        if graph_token_idx is not None:
            _inputs['graph_token_idx'] = graph_token_idx
        
        return _inputs

AutoConfig.register("llava_mora", LlavaMoRAConfig)
AutoModelForCausalLM.register(LlavaMoRAConfig, LlavaLlamaMoRAForCausalLM)