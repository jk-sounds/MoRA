#!/bin/bash

#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1"
PROMPT_VERSION=v1
MODEL_VERSION=../vicuna-7b/
GRAPH_TOWER="moleculestm"
INIT_CHECKPOINT_GNN="../MoleculeSTM/molecule_model.pth"
CHECKPOINT_FOLDER_PREFIX="../all_checkpoints/MoRA"
TASK="molcap"
deepspeed --include="localhost:0,1,2,3,4,5,6,7" llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable False\
    --model_name_or_path $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path "../MolInstruction/molecular_description_train.json" \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-mora-$GRAPH_TOWER \
    --num_train_epochs 30 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type mora \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none \
    --evaluation_strategy no \
    --mora_depth 8 \
    --mora_llm_depth 32 \
    --mora_rank 64 \
    --mora_alpha 64 \
    --mora_type qkvom \
    --weights_sep True \
    --skip_layers 4

