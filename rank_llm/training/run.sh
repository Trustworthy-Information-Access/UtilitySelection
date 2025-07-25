#!/bin/bash

BASE_MODEL="model/models/Qwen/Qwen3-1.7B"
TRAIN_DATA_PATH="results/utility_answer_both_final.jsonl"  # Train Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="output/annotation_utility_answer_both"  # Directory to save the trained model

mkdir -p "${OUTPUT_DIR}"
 MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 \
    --config_file "configs/accel_config_deepspeed.yaml" \
    train_rankllm.py \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --num_train_epochs 3 \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 50 \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --noisy_embedding_alpha 5 \
    --objective generation \
    --with_tracking \
    --report_to wandb \
    --checkpointing_steps epoch
