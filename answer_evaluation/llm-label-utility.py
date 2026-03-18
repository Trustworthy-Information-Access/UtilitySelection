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
def collate_fn(batch, tokenizer):
    utility_prompt = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    formated_passages = [item[2] for item in batch]
    formated_passages_ids = [item[3] for item in batch]
    query_id = [item[4] for item in batch]
    inputs = tokenizer.apply_chat_template(utility_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return inputs, formated_passages, formated_passages_ids, query_id
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
    if 'llama' in model_args.model_name_or_path:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    # 检查文件是否存在
    if os.path.exists(data_args.output_dir):
        # 计算文件当前行数
        with open(data_args.output_dir, "r", encoding="utf-8") as file_r:
            line_count = sum(1 for _ in file_r)
            if line_count > 100:
                print(f"文件已存在且行数({line_count})匹配，跳过处理")
                exit()  # 或 return 根据上下文环境选择
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = EncodeDataset(data_args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))
    file_w = open(data_args.output_dir, "w", encoding="utf-8")
    llm = LLM(
        model=model_args.model_name_or_path,
        dtype=torch.float16,
        tensor_parallel_size=8
    )
    for utility_prompts, formated_passageses, formated_passages_idses, query_ids in dataloader:
        outputs = llm.generate(utility_prompts, sampling_params)
        ress = []
        for output in outputs:
            res = output.outputs[0].text
            ress.append(res)

        for i in range(len(query_ids)):
            js = {}
            js["query_id"] = query_ids[i]
            # js["passages"] = formated_passageses[i]
            js["passages_ids"] = formated_passages_idses[i]
            js["answer"] = ress[i]
            # js["relevance_output"] = ress_relevance[i]
            file_w.write(json.dumps(js)+"\n")
    

        



    