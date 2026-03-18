#!/bin/bash

for data in "hotpotqa_100";
do 
    for topk in 5 10 15 20 40 60 80 100;
    do 
    model_name_or_path="model/llama31-8b-instruct"
    dataset_path="output/Qwen3-1.7B/results_bge/"$data".jsonl"
    output_dir="output/Qwen3-1.7B/results_bge/llama31_"$data"-answer_"$topk"_nw0.jsonl"
    echo $dataset_path
    python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path  --batch_size 512 --output_dir $output_dir --topk $topk --nw False
    sleep 2m
    done
done
