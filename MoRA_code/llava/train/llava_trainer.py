import os
import torch

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":  # ���ƫ������Ϊ none
        to_return = {k: t for k, t in named_params if "lora_" in k}  # �����ذ��� "lora_" �Ĳ���
    elif bias == "all":  # ���ƫ������Ϊ all
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}  # ���ذ��� "lora_" �� "bias" �Ĳ���
    elif bias == "lora_only":  # ���ƫ������Ϊ lora_only
        to_return = {}  # ��ʼ�������ֵ�
        maybe_lora_bias = {}  # ��ʼ�����ܵ� LoRA ƫ���ֵ�
        lora_bias_names = set()  # ��ʼ�� LoRA ƫ�����Ƽ���
        for k, t in named_params:  # �������в���
            if "lora_" in k:  # ����������ư��� "lora_"
                to_return[k] = t  # ���ӵ������ֵ�
                bias_name = k.split("lora_")[0] + "bias"  # ����ƫ������
                lora_bias_names.add(bias_name)  # ���ӵ�ƫ�����Ƽ���
            elif "bias" in k:  # ����������ư��� "bias"
                maybe_lora_bias[k] = t  # ���ӵ����ܵ� LoRA ƫ���ֵ�
        for k, t in maybe_lora_bias:  # �������ܵ� LoRA ƫ���ֵ�
            if bias_name in lora_bias_names:  # ���ƫ�������ڼ�����
                to_return[bias_name] = t  # ���ӵ������ֵ�
    else:
        raise NotImplementedError  # �׳�δʵ�ִ���
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}  # ����ÿ������
    return to_return  # ���ش�����Ĳ���
    
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}  # ���ز����� "lora_" �Ĳ���
    if require_grad_only:  # �������Ҫ��ѵ������
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}  # ���˳���ѵ������
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}  # ����ÿ������
    return to_return  # ���ش�����Ĳ���

class LLaVATrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        state_dict = get_peft_state_maybe_zero_3(
            self.model.named_parameters(), self.args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters()
        )
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            print(f"save models to {output_dir} ")
            self.model.config.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)