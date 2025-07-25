import json
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
with open("utility_answer_both_final.jsonl", "w", encoding="utf-8") as file_w:
    with open("utility_answer_both.jsonl", "r", encoding="utf-8") as file_r:
        for line in file_r:
            js = json.loads(line)
            query = js["query"]
            query_id = js["query_id"]
            passages = js["passages"]
            utility_output = js["utility_output"]
            if "Answer:" in utility_output and "selection:" in utility_output:
                messages = get_direct_judge_list_utility(query,  passages)
                file_w.write(json.dumps({"query_id": query_id, "messages": messages, "output": utility_output})+"\n")
