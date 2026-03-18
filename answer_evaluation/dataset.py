import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple
from arguments import DataArguments
def format_query(query: str) -> str:
    return f'{query.strip()}'.strip()

def format_passage(text: str, title: str = '') -> str:
    # if len(text.split(" ")) > 300:
    #     text = " ".join(text.split(" "))
    return f'{title.strip()} {text.strip()}'.strip()

def get_prefix_direct_judge_list(query, num):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \n I will also provide you with a reference answer to the question. \nSelect the passages that have utility in generating the reference answer to the following question from the {num} passages: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]
def get_post_direct_judge_list(query, instruct, answer):
    return f"Question: {query}. \n Reference answer: {answer}. \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct
def get_direct_judge_list_utility(question, instruct, passages, answer):
    messages = get_prefix_direct_judge_list(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list(question, instruct, answer)})
    return messages
# def get_prefix_direct_judge_list_relevance(query, num):
#     return [{'role': 'user',
#              'content': "You are the relevance judger, an intelligent assistant that can select the passages that relevant to the question."},
#             {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
#             {'role': 'user',
#              'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages thet are relevant to the question: {query}."},
#             {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

def get_prefix_direct_judge_list_relevance(query, num):
    return [{'role': 'user',
             'content': "You are the relevance judger, an intelligent assistant that can select the passages that relevant to the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with 16 passages, each indicated by number identifier []. Rank them based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_relevance(query, instruct):
    return f"Search Query: {query}."+instruct
def get_direct_judge_list_relevance(question, instruct, passages):
    messages = get_prefix_direct_judge_list_relevance(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_relevance(question, instruct)})
    return messages
def generate_answer_prompt_passages(question, passages, nw):
    pas = '\n'.join(passages)
    if nw: 
        return [
            {'role': 'user', 'content': f"Question: {question} \n Passages: {pas} \n  Answer the question based on the given passages  or your internal knowledge with one or few words without the explanation. Answer: "},]
    else:
        return [
            {'role': 'user', 'content': f"Question: {question} \n Passages: {pas} \n  Answer the question based on the given passages with one or few words without the explanation. Answer: "},]



def generate_answer_prompt_passages_sentence(question, passages, nw):
    pas = '\n'.join(passages)
    if nw:
        return [
            {'role': 'user', 'content': f"Question: {question} \n Passages: {pas} \n  Answer the question based on the given passages  or your internal knowledge with one or few sentences without the explanation. Answer: "},]
    else:
        return [
            {'role': 'user', 'content': f"Question: {question} \n Passages: {pas} \n  Answer the question based on the given passages with one or few sentences without the explanation. Answer: "},]

class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
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
        query = group['query']
        query_id = group['query_id']
        labels = []
        formated_passages = []
        formated_passages_ids = []
        if "passages" in group: 
            passages = group['passages'][:self.data_args.topk]
        else:
            passages = group['hits'][:self.data_args.topk]
        # formated_passages = passages
        for passage in passages:
            if 'contents' in passage:
                formated_passages.append(passage['contents'])
                formated_passages_ids.append(passage['id'])
            elif 'content' in passage:
                formated_passages.append(passage['content'])
                formated_passages_ids.append(passage['docid'])
            elif "title" in passage and 'text' in passage:
                formated_passages.append(format_passage(passage['text'], passage['title']))
                formated_passages_ids.append(passage['_id'])
            else:
                assert 1 > 2
        # if 'contents' in group['passages'][0]:
        #     formated_passages = [format_passage(passage['contents']) for passage in group['passages'][:self.data_args.topk]]
        #     formated_passages_ids = [passage['id'] for passage in group['passages'][:self.data_args.topk]]
        # else:
        #     formated_passages = [format_passage(passage['text'], passage['title']) for passage in group['passages'][:self.data_args.topk]]
        #     formated_passages_ids = [passage['_id'] for passage in group['passages'][:self.data_args.topk]]
        labels = [0]*len(formated_passages)
        if "msmarco" in self.data_args.dataset_path:
            messages = generate_answer_prompt_passages_sentence(query, formated_passages, self.data_args.nw)
        else:
            messages = generate_answer_prompt_passages(query, formated_passages, self.data_args.nw)
        utility_prompt = messages
        return utility_prompt, labels, formated_passages, formated_passages_ids, query_id