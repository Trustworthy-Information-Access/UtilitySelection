
model_path="results/models/annotation_utility_answer_both/"
output_path="results/results_first10_merge_remian/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python run_evaluation_utility_selection.py --top_k 100 --output_path $output_path --model_path $model_path

model_path="results/models/annotation_utility_answer_both/"
output_path="results/results_first10_merge_remian_bge/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python run_evaluation_utility_selection_bge.py --top_k 100 --output_path $output_path --model_path $model_path


model_path="results/models/Qwen3-32B_annotation_relevance_rankingr/"
output_path="results/models/Qwen3-32B_annotation_relevance_ranking/results/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python run_evaluation.py --top_k 100 --output_path $output_path --model_path $model_path


model_path="results/models/Qwen3-32B_annotation_relevance_ranking/"
output_path="results/models/Qwen3-32B_annotation_relevance_ranking/results/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python run_evaluation_hotpotqa.py --top_k 100 --output_path $output_path --model_path $model_path

