import copy
import time
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'query_id': qid,'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_direct_judge_list_utility(query, num):
    return [{'role': 'system',
             'content': "You are the utility selector, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility selector.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that have utility in answering the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_utility(query, num):
    return f"Question: {query}\n Utility Requirement: A passage has **utility** only if it is **both relevant to the question AND useful for generating a correct and reasonable answer.**\n Firstly,  **provide the answer** to the question based on the provided {num} passages or your own knowledge without sources, which can help judge passages utility. \n Then,  **select the passages that have utility in answering the question**. \n**Output Format STRICTLY:**\nAnswer: [Your generated answer here]\nMy selection: [[i],[j],...]\nOnly respond to the answer and the selected list, do not say any word or explain."

# def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
#     query = item['query']
#     num = len(item['hits'][rank_start: rank_end])
#     max_length = 300
#     messages = get_prefix_direct_judge_list_utility(query, num)
#     rank = 0
#     for hit in item['hits'][rank_start: rank_end]:
#         rank += 1
#         content = hit['content']
#         content = content.replace('Title: Content: ', '')
#         content = content.strip()
#         # For Japanese should cut by character: content = content[:int(max_length)]
#         content = ' '.join(content.split()[:int(max_length)])
#         messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
#         messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
#     messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility(query, num)})
#     return messages
def create_window_instruction(query, window_passages):
    num = len(window_passages)
    max_length = 300
    messages = get_prefix_direct_judge_list_utility(query, num)
    for idx, passage in enumerate(window_passages, 1):
        content = passage['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{idx}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{idx}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility(query, num)})
    return messages


def run_llm(messages, llm, tokenizer):
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    outputs = llm.generate(prompts, sampling_params)
    response = outputs[0].outputs[0].text
    print(response)
    return response

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def parse_selection_response(response):
    if 'selection:' in response:
        selection_line = response.split("selection:")[1]
    else:
        selection_line = response.split("selection:")
    if not isinstance(selection_line, str) or not selection_line:
        matches = []
    else:
        response = re.findall(r'\[(\d+)\]', selection_line)
        matches = [int(x) - 1 for x in response]
    return remove_duplicate(matches)

def utility_selection(llm, tokenizer, window_passages, query):
    messages = create_window_instruction(query, window_passages)
    response = run_llm(messages, llm, tokenizer)
    return parse_selection_response(response)

def utility_selection_pipeline(llm, tokenizer, item, window_size):
    all_passages = item['hits']
    query = item['query']
    selected = []
    pointer = 0
    n_passages = len(all_passages) #100
    # Process initial window
    if pointer < n_passages:
        window = all_passages[pointer:pointer+window_size]
        pointer += window_size
        selected_indices = utility_selection(llm, tokenizer, window, query)
        for idx in selected_indices:
            if idx < len(window):
                selected.append(window[idx])
    # Process subsequent windows
    while pointer < n_passages:
        print([passage["docid"] for passage in selected])
        print("selected above----------------")
        last_selected = selected[:10] 
        # last_selected = selected[-5:] if len(selected) > 5 else selected
        # selected = selected[:-len(last_selected)] if len(selected) >= 5 else []
        n_new = window_size - len(last_selected)
        if n_new <= 0:
            break
        new_passages = all_passages[pointer:pointer+n_new]
        pointer += n_new
        if not new_passages:
            break
        window = last_selected + new_passages
        print("new continue window: ", [passage["docid"] for passage in window])
        selected_indices = utility_selection(llm, tokenizer, window, query)
        print("selected_ids: ", selected_indices)
        new_selected = []
        for idx in selected_indices:
            if idx < len(window) and idx >=0:
                new_selected.append(window[idx])
            # if idx < len(window) and idx >= len(last_selected):
            #     new_selected.append(window[idx])
        selected = new_selected + [doc for doc in selected if doc not in new_selected]
        # if len(selected)>0:
        #     selected = new_selected + selected[10:]
        # else:
        #     selected = new_selected
    print("final_selected: ", [passage["docid"] for passage in selected])
    item['hits'] = selected
    return item

def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

