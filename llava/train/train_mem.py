import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llava.train.llama_mora_flash_attn_monkey_patch import replace_llama_mora_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
replace_llama_mora_attn_with_flash_attn()

from llava.train.train_drug import train

if __name__ == "__main__":
    train()