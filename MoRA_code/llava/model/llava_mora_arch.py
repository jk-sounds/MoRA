#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch

import torch.nn as nn

from .multimodal_encoder.builder import build_graph_tower
from .MoRA.weights_generater import get_lora_generater
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LlavaMoRAMetaModel:

    def __init__(self, config):
        print("__init__ begin", flush=True)
        super(LlavaMoRAMetaModel, self).__init__(config)

        if hasattr(config, "mm_graph_tower"):
            print(f"Config has mm_graph_tower, will create graph_tower and mm_projector")
            self.graph_tower = build_graph_tower(config, delay_load=True)
            self.mm_projector = get_lora_generater(config)


    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, model_args, fsdp=None):
        graph_tower = model_args.graph_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_graph_tower = graph_tower

        if self.get_graph_tower() is None:
            graph_tower = build_graph_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = graph_tower.hidden_size
        # MoRA specific configurations
        self.config.mora_dim = model_args.mora_dim
        self.config.mora_depth = model_args.mora_depth
        self.config.mora_visual_dim = model_args.mora_visual_dim
        self.config.mora_pos_num = model_args.mora_pos_num
        self.config.mora_llm_dim = model_args.mora_llm_dim
        self.config.mora_llm_depth = model_args.mora_llm_depth
        self.config.mora_rank = model_args.mora_rank
        self.config.mora_type = model_args.mora_type
        self.config.mora_alpha = model_args.mora_alpha
        self.config.weights_sep = model_args.weights_sep
        self.config.skip_layers = model_args.skip_layers

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = get_lora_generater(model_args)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            missing_keys, unexpected_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print(f"missing_keys is {missing_keys} \n unexpected_keys is {unexpected_keys}")

            if hasattr(self, 'layer_attention_scorer'):
                self.layer_attention_scorer.load_state_dict(get_w(mm_projector_weights, 'layer_attention_scorer'))
                self.layer_attention_transform.load_state_dict(get_w(mm_projector_weights, 'layer_attention_transform'))
                print("\nload layer_attention\n", flush=True)

            if hasattr(self, 'fpn_head_module'):
                fpn_head_weights = get_w(mm_projector_weights, 'fpn_head_module')
                self.fpn_head_module.load_state_dict(fpn_head_weights)
                print("\nload fpn_head_module\n", flush=True)


class LlavaGraphMoRAMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def extract_selfies(self, input_ids):
        selfies_ids = []
        for i in range(input_ids):
            if input_ids[i] >= 32000:
                selfies_ids.append(input_ids[i])
        return selfies_ids


    def encode_graphs(self, images):
        # Handle Batch objects from training data collator
        from torch_geometric.data import Batch
        if isinstance(images, Batch):
            # For Batch objects, pass directly to graph tower
            graph_output = self.get_graph_tower()(images)
        elif isinstance(images, list):
            # For list of Data objects (evaluation case)
            if len(images) == 1:
                graph_output = self.get_graph_tower()(images[0])
            else:
                # Create batch from list
                batch = Batch.from_data_list(images)
                graph_output = self.get_graph_tower()(batch)
        else:
            # Single Data object
            graph_output = self.get_graph_tower()(images)
        
        # Extract features from tuple output
        if isinstance(graph_output, tuple):
            # Use the second element (node features) based on encode_mol_ pattern
            image_features = graph_output[1] if len(graph_output) > 1 else graph_output[0]
            
            # If it's a list of features from multiple layers, use the last one
            if isinstance(image_features, list):
                image_features = image_features[-1]
        else:
            image_features = graph_output
        
        # Ensure the features have 3 dimensions [batch_size, seq_len, feature_dim]
        if image_features.dim() == 2:
            # Add batch dimension if missing
            image_features = image_features.unsqueeze(0)
        elif image_features.dim() == 1:
            # Add both batch and sequence dimensions
            image_features = image_features.unsqueeze(0).unsqueeze(0)
        elif image_features.dim() > 3:
            # If more than 3 dimensions, try to reshape
            # This might happen with graph batches
            batch_size = image_features.shape[0]
            image_features = image_features.view(batch_size, -1, image_features.shape[-1])
        
        if torch.isnan(image_features).any():
               raise ValueError("image_features,FATAL: NaNs detected in the final hidden_states of the model.")
        
        mora_weights = self.get_model().mm_projector(image_features)
        return mora_weights

    def encode_mol_(self, mol, input_ids, labels=None):
        main_device = self.get_model().base_model.device
        _, h_node = self.get_model().get_graph_tower().encode_mol(mol, proj=False, return_node_feats=True)
        h_list_on_device = [h.to(main_device) for h in h_node]
        weighted_h_node = h_list_on_device[-1]
        dtype = weighted_h_node.dtype
        
        # Generate mora weights instead of projecting features
        if dtype == torch.bfloat16:
            weighted_h_node = weighted_h_node.float()
            mora_weights = self.get_model().mm_projector.float()(weighted_h_node).to(dtype=dtype)
        else:
            mora_weights = self.get_model().mm_projector(weighted_h_node)

        all_text_embedding = self.get_input_embeddings()(input_ids.clamp(min=0)).detach()
        final_text_embedding = all_text_embedding.squeeze(0)
        return mora_weights, final_text_embedding


    def initialize_graph_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False