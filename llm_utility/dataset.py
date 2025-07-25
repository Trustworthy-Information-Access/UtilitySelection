import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple
from arguments import DataArguments
import json
import re
#####################################################################
# def get_direct_judge_list_utility_ranking(query, passages, answer):
#     num = len(passages)
#     messages = get_prefix_direct_judge_list_utility_ranking(query, num, answer)
#     rank = 0
#     for passage in passages:
#         rank += 1
#         content = passage
#         messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
#         messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
#     messages.append({'role': 'user', 'content': get_post_prompt_utility_ranking(query, num, answer)})
#     return messages

# def get_prefix_direct_judge_list_utility_ranking(query, num, answer):
#     return [{'role': 'user',
#             'content': "You are the utility ranker, an intelligent assistant that can rank passages based on their utility in generating the given reference answer to the question."},
#             {'role': 'assistant',
#             'content': "Yes, i am the ranker."},
#             {'role': 'user',
#             'content': f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the question. \nRank the passages based on their utility in generating the reference answer to the question: {query}."},
#             {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]

# def get_post_prompt_utility_ranking(query, num, answer):
#     return f"Question: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their utility in generating the reference answer to the question. The passages should be listed in utility descending order using identifiers.  The passages that have utility generating the reference answer to the question should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."

def get_prefix_direct_judge_list_utility(query, num):
    return [{'role': 'system',
             'content': "You are the utility selector, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility selector.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that have utility in answering the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_utility(query, num):
    return f"Question: {query}\n Utility Requirement: A passage has **utility** only if it is **both relevant to the question AND useful for generating a correct and reasonable answer.**\n Firstly,  **provide the answer** to the question based on the provided {num} passages or your internal knowledge without sources, which can help you judge passages' utility. \n Then,  **select the passages that have utility in answering the question**. \n**Output Format STRICTLY:**\nAnswer: [Your generated answer here]\nMy selection: [[i],[j],...]\nOnly respond to the answer and the selected list, do not say any word or explain."
def get_direct_judge_list_utility(question, passages):
    messages = get_prefix_direct_judge_list_utility(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility(question, len(passages))})
    return messages
#####################################################################


def format_query(query: str) -> str:
    return f'{query.strip()}'.strip()

def format_passage(text: str, title: str = '') -> str:
    return f'{title.strip()} {text.strip()}'.strip()
#####################################################################
def get_prefix_direct_judge_list_relevance(query, num):
    return [{'role': 'user',
             'content': "You are the relevance ranker, an intelligent assistant that can rank the passages based on their relevance to the search query."},
            {'role': 'assistant', 'content': 'Yes, i am the relevance ranker.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to the search query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_relevance(query, num):
    return f"Search Query: {query}. Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

# def get_prefix_direct_judge_list_relevance_selection(query, num):
#     return [{'role': 'user',
#              'content': "You are the relevance judger, an intelligent assistant that can select the passages that relevant to the question."},
#             {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
#             {'role': 'user',
#              'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages thet are relevant to the question: {query}."},
#             {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
# def get_post_direct_judge_list_relevance_selection(query, num):
#     return f"Search Query: {query}. Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. "

def get_direct_judge_list_relevance(question, passages):
    messages = get_prefix_direct_judge_list_relevance(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_relevance(question, len(passages))})
    return messages

# def get_direct_judge_list_relevance_selection(question, passages):
#     messages = get_prefix_direct_judge_list_relevance_selection(question, len(passages))
#     rank = 0
#     for content in passages:
#         rank += 1
#         messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
#         messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
#     messages.append({'role': 'user', 'content': get_post_direct_judge_list_relevance_selection(question, len(passages))})
#     return messages


def generate_answer_prompt_passages(question, passages, nw_knowledge):
    pas = '\n'.join(passages)
    if nw_knowledge:
        return [
                {'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information or your internal knowledge with one or a few sentences without the source.\n Question: {question}\n\n Answer:"},]
    else:
        return [
                {'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information with one or a few sentences without the source.\n Question: {question}\n\n Answer:"},]

class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.utility_instruct = """
        Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        labels = []
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["negative_passages"]
        # if "relevance" in self.data_args.type_prompt or self.data_args.type_prompt == "answer":
        #     group_negatives = group["negative_passages"]
        # elif self.data_args.type_prompt == "utility":
        #     group_negatives = group["passages"]
        #     assert len(group_negatives) == len(group["passages_ids"])
        #     answer = group["answer_output"]
        #     if "</think>" in answer and "<think>" in answer:
        #         answer = answer.split("</think>")[-1]
        # elif self.data_args.type_prompt == "answer":
        #     relevance_selection = group["relevance_selection"]
        #     numbers = re.findall(r'\[(\d+)\]', relevance_selection)
        #     numbers = [int(number)-1 for number in numbers]
        #     group_negatives = [group["passages"][index] for index in numbers]

        for index, passage in enumerate(group_negatives):
            formated_passages.append(passage["text"])
            formated_passages_ids.append(passage["docid"])
            # if self.data_args.type_prompt == "utility": 
            #     formated_passages.append(passage)
            # else:
            #     formated_passages.append(passage["text"])
            # if self.data_args.type_prompt == "utility":
            #     formated_passages_ids.append(group["passages_ids"][index])
            # else:
            #     formated_passages_ids.append(passage["docid"])
        assert self.data_args.type_prompt in ["answer", "utility", "relevance_selection",  "relevance_ranking"]
        if self.data_args.type_prompt == "answer":
            messages = generate_answer_prompt_passages(query, formated_passages, self.data_args.nw_knowledge)
        elif self.data_args.type_prompt == "utility":
            # messages = get_direct_judge_list_utility_ranking(query, formated_passages, answer)
            messages = get_direct_judge_list_utility(query, formated_passages)
        elif self.data_args.type_prompt == "relevance_ranking":
            messages = get_direct_judge_list_relevance(query,  formated_passages)

           
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.thinking)
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, labels, formated_passages, formated_passages_ids, query_id, query