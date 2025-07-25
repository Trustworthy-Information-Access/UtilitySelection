import copy
import time
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams


def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


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
            ranks.append({'query': query, 'hits': []})
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




def get_prefix_direct_judge_list_utility_ranking(query, num, answer):
    return [{'role': 'user',
            'content': "You are the utility ranker, an intelligent assistant that can rank passages based on their utility in generating the given reference answer to the question."},
            {'role': 'assistant',
            'content': "Yes, i am the ranker."},
            {'role': 'user',
            'content': f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the question. \nRank the passages based on their utility in generating the reference answer to the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]

def get_post_prompt_utility_ranking(query, num, answer):
    return f"Question: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their utility in generating the reference answer to the question. The passages should be listed in utility descending order using identifiers.  The passages that have utility generating the reference answer to the question should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."


def get_prefix_direct_judge_list_relevance(query, num):
    return [{'role': 'user',
             'content': "You are the relevance ranker, an intelligent assistant that can rank the passages based on their relevance to the search query."},
            {'role': 'assistant', 'content': 'Yes, i am the relevance ranker.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to the search query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_relevance(query, num):
    return f"Search Query: {query}. Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."


def generate_answer_prompt_passages(question, passages):
    pas = '\n'.join(passages)
    return [{'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information with one or a few sentences without the source.\n Question: {question}\n\n Answer:"},]

def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_direct_judge_list_relevance(query, num)
    # messages = get_prefix_direct_judge_list_utility_ranking(query, num, answer)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_relevance(query, num)})
    # messages.append({'role': 'user', 'content': get_post_prompt_utility_ranking(query, num)})

    return messages


def run_llm(messages, llm, tokenizer):
    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    outputs = llm.generate(prompts, sampling_params)
    response = outputs[0].outputs[0].text
    print(response)
    return response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(llm, tokenizer, item=None, rank_start=0, rank_end=100):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)  # chan
    permutation = run_llm(messages, llm, tokenizer)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(llm, tokenizer, item=None, rank_start=0, rank_end=100, window_size=20, step=10):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    answer_flag = 0
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(llm, tokenizer, item, start_pos, end_pos)
        end_pos = end_pos - step
        start_pos = start_pos - step
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
