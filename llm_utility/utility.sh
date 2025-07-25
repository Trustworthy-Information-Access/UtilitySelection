

sleep 1m
export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_name_or_path="model/models/Qwen/Qwen3-32B"
dataset_path="datasets/marco-train-100k.jsonl"
output_dir="results/utility_answer_both.jsonl"
train_group_size=16
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir --thinking False --type_prompt utility




