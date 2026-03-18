

export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_name_or_path="models/Qwen/Qwen3-32B"
dataset_path="sigir-ap/marco-train-100k.jsonl"
output_dir="sigir-ap/annotation_relevance_ranking.jsonl"
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 4096 --output_dir $output_dir --thinking False --type_prompt relevance_ranking

dataset_path="sigir-ap/marco-train-100k.jsonl"
output_dir="sigir-ap/annotation_answer_nw1.jsonl"
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 4096 --output_dir $output_dir --thinking False --type_prompt answer --nw_knowledge True

dataset_path="sigir-ap/annotation_answer_nw1.jsonl"
output_dir="sigir-ap/annotation_utility_nw1_def.jsonl"
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 4096 --output_dir $output_dir --thinking False --type_prompt utility
