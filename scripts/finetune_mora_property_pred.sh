#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="../graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="../MoleculeSTM/molecule_model.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="../all_checkpoints/MoRA"
TASK="property_pred"

deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable False \
    --model_name_or_path ../$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ../MolInstruction/property_prediction_train.json \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type mora \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-mora-$GRAPH_TOWER \
    --num_train_epochs 10 \
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
