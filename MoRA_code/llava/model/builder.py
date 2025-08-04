import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def update_pretrained_config(pretrained_config, update_config):
    if isinstance(pretrained_config, dict):
        pretrained_config.update(update_config)
    else:
        config_class = pretrained_config.__class__
        cfg_pretrained_dict = pretrained_config.to_dict()
        cfg_pretrained_dict.update(**update_config)
        pretrained_config = config_class(**cfg_pretrained_dict)
    return pretrained_config

def remove_base_layer_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if ".base_layer." in key:
            new_key = key.replace(".base_layer", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
    
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, mm_encoder_cfg=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        # kwargs['load_in_8bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["mm_projector", "compressor"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'mora' in model_name.lower() and model_base is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if mm_encoder_cfg is not None:
                cfg_pretrained = update_pretrained_config(cfg_pretrained, mm_encoder_cfg)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            # currently changed to LlavaGraphLlamaForCausalLM
            model = LlavaLlamaMoRAForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                               config=cfg_pretrained, **kwargs)
            model.initialize_graph_tokenizer(cfg_pretrained, tokenizer)
            model.config = cfg_pretrained
            #model.base_model.graph_tower.select_feature = cfg_pretrained.mm_graph_select_feature
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                print('model.lm_head.weight.shape[0] != token_num',flush=True)
            
            print('Loading additional LLaVA weights...')
            for name, param in model.named_parameters():
                print(name,flush=True)
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                                 map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')

                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                   non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                       non_lora_trainables.items()}
            non_lora_trainables = remove_base_layer_prefix(non_lora_trainables)
            non_lora_trainables = {k: v.to('cuda:0') for k, v in non_lora_trainables.items()} 
            print("*"*50,flush=True)
            for name in non_lora_trainables.keys():
                print(name,flush=True)
            #model.load_state_dict(non_lora_trainables, strict=False)
            print('\n------no LoAR para---------\n',flush=True)
            missing_keys, unexpected_keys = model.load_state_dict(non_lora_trainables, strict=False)
            print(f"miss para: {missing_keys}",flush=True)  
            print(f"unexpect para: {unexpected_keys}",flush=True) 
            print('Model is loaded...')

            unimol_prefix = "model.graph_tower."
            ckpt = torch.load('../MoleculeSTM/molecule_model.pth', map_location='cpu', weights_only=True)
 
            if 'model' in ckpt:
                unimol_state_dict = ckpt['model']
            else:
                print("Warning: 'model' key not found in checkpoint. Assuming the entire file is the state_dict.")
                unimol_state_dict = ckpt
            
            print(f"Extracted UniMol state_dict with {len(unimol_state_dict)} parameters.")

            prefixed_state_dict = {}
            for key, value in unimol_state_dict.items():
                new_key = unimol_prefix + key
                prefixed_state_dict[new_key] = value

            example_original_key = list(unimol_state_dict.keys())[0]
            example_new_key = list(prefixed_state_dict.keys())[0]
            print(f"Example key mapping: '{example_original_key}' -> '{example_new_key}'",flush=True)

            print("Loading prefixed state_dict into the main model...",flush=True)
            missing_keys, unexpected_keys = model.load_state_dict(prefixed_state_dict, strict=False)
            print(f"miss para: {missing_keys}",flush=True) 
            print(f"unexpect para: {unexpected_keys}",flush=True) 

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'),
                                    os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMiniLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)

            model.base_model.build_compressor(cfg_pretrained)
            model.initialize_vision_tokenizer(cfg_pretrained, tokenizer)
            model.config = cfg_pretrained

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaMiniLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                                             **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len



