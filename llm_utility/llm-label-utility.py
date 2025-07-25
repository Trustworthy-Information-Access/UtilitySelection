import sys
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
os.environ["WORLD_SIZE"] = "8"
os.environ['MASTER_PORT'] = '29203'
import logging
import torch
import transformers
import json
#dataset
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from dataset import EncodeDataset
from arguments import ModelArguments, DataArguments
from transformers import (
    HfArgumentParser,
)
def collate_fn(batch):
    prompt = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    formated_passages = [item[2] for item in batch]
    formated_passages_ids = [item[3] for item in batch]
    query_id = [item[4] for item in batch]
    query = [item[5] for item in batch]
    return prompt, labels, formated_passages, formated_passages_ids, query_id, query
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
    # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768)
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=8196)
    sampling_params = SamplingParams(temperature=0, max_tokens=4096)
    # Temperature=0.6、TopP=0.95、TopK=20 和 MinP=0SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

    # sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    llm = LLM(
        model=model_args.model_name_or_path,
        dtype=torch.float16,
        # gpu_memory_utilization=0.8,
        tensor_parallel_size=8
    )
    # llm = LLM(model=model_args.model_name_or_path, tensor_parallel_size=8)
    dataset = EncodeDataset(data_args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=collate_fn)
    file_w = open(data_args.output_dir, "w", encoding="utf-8")
    for prompts, labels, formated_passageses, formated_passages_idses, query_ids, queries in dataloader:
        outputs = llm.generate(prompts, sampling_params)
        ress = []
        for output in outputs:
            res = output.outputs[0].text
            ress.append(res)
        for i in range(len(query_ids)):
            js = {}
            js["query_id"] = query_ids[i]
            js["passages"] = formated_passageses[i]
            js["passages_ids"] = formated_passages_idses[i]
            js["query"] = queries[i]
            if data_args.type_prompt == "answer":
                js["answer_output"] = ress[i]
            elif data_args.type_prompt == "utility":
                js["utility_output"] = ress[i]
            elif data_args.type_prompt == "relevance_ranking":
                js["relevance_ranking"] = ress[i]
            elif data_args.type_prompt == "relevance_selection":
                js["relevance_selection"] = ress[i]
            file_w.write(json.dumps(js)+"\n")
