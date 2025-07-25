import json
# test0 0 doc0 1
from collections import defaultdict
query_pos = defaultdict(list)
with open("qrels.beir-v1.0.0-hotpotqa.test.txt", "r", encoding="utf-8") as file_r:
    for line in file_r:
        query_id, _, docid, _ = line.strip().split(" ")
        query_pos[query_id].append(docid)
recalls = []
precisions = []
for rank in [5, 10, 15, 20, 40, 60, 80, 100]:
    with open("results_bge/hotpotqa_100.jsonl", "r", encoding="utf-8") as file_r:
        for line in file_r:
            js = json.loads(line)
            query_id = js["query_id"]
            docids = [passage["docid"] for passage in js["hits"][:rank]]
            positive = query_pos[query_id]
            sam_positive = [1 for docid in docids if docid in positive]
            recall = sum(sam_positive)/len(positive) if  len(positive) > 0 else 0
            precision = sum(sam_positive)/len(docids) if  len(docids) > 0 else 0
            recalls.append(recall)
            precisions.append(precision)
    all_recall = sum(recalls)/len(recalls)
    all_precision = sum(precisions)/len(precisions)
    print(all_recall, " ", all_precision, " ", 2 * (all_precision * all_recall) / (all_precision + all_recall))
