import argparse  # 添加argparse模块用于解析命令行参数
bm25_index = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    # 'dl21': 'msmarco-v2-passage-dev',
    # 'dl22': 'msmarco-v2-passage-dev',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',
    'nq': 'beir-v1.0.0-nq.flat',
    'hotpotqa': 'beir-v1.0.0-hotpotqa.flat',
    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}
THE_INDEX = {
    'dl19': 'msmarco-v1-passage.bge-base-en-v1.5.hnsw',
    'dl20': 'msmarco-v1-passage.bge-base-en-v1.5.hnsw',
    'covid': 'beir-v1.0.0-trec-covid.bge-base-en-v1.5.hnsw',
    'arguana': 'beir-v1.0.0-arguana.bge-base-en-v1.5.hnsw',
    'touche': 'beir-v1.0.0-webis-touche2020.bge-base-en-v1.5.hnsw',
    'news': 'beir-v1.0.0-trec-news.bge-base-en-v1.5.hnsw',
    'scifact': 'beir-v1.0.0-scifact.bge-base-en-v1.5.hnsw',
    'fiqa': 'beir-v1.0.0-fiqa.bge-base-en-v1.5.hnsw',
    'scidocs': 'beir-v1.0.0-scidocs.bge-base-en-v1.5.hnsw',
    'nfc': 'beir-v1.0.0-nfcorpus.bge-base-en-v1.5.hnsw',
    'quora': 'beir-v1.0.0-quora.bge-base-en-v1.5.hnsw',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.bge-base-en-v1.5.hnsw',
    'fever': 'beir-v1.0.0-fever.bge-base-en-v1.5.hnsw',
    'robust04': 'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw',
    'signal': 'beir-v1.0.0-signal1m.bge-base-en-v1.5.hnsw',
    'nq': 'beir-v1.0.0-nq.bge-base-en-v1.5.hnsw',
    'hotpotqa': 'beir-v1.0.0-hotpotqa.bge-base-en-v1.5.hnsw',
    
    # 'dl19': 'msmarco-v1-passage.bge-base-en-v1.5',
    # 'dl20': 'msmarco-v1-passage.bge-base-en-v1.5',
    # 'covid': 'beir-v1.0.0-trec-covid.bge-base-en-v1.5',
    # 'arguana': 'beir-v1.0.0-arguana.bge-base-en-v1.5',
    # 'touche': 'beir-v1.0.0-webis-touche2020.bge-base-en-v1.5',
    # 'news': 'beir-v1.0.0-trec-news.bge-base-en-v1.5',
    # 'scifact': 'beir-v1.0.0-scifact.bge-base-en-v1.5',
    # 'fiqa': 'beir-v1.0.0-fiqa.bge-base-en-v1.5',
    # 'scidocs': 'beir-v1.0.0-scidocs.bge-base-en-v1.5',
    # 'nfc': 'beir-v1.0.0-nfcorpus.bge-base-en-v1.5',
    # 'quora': 'beir-v1.0.0-quora.bge-base-en-v1.5',
    # 'dbpedia': 'beir-v1.0.0-dbpedia-entity.bge-base-en-v1.5',
    # 'fever': 'beir-v1.0.0-fever.bge-base-en-v1.5',
    # 'robust04': 'beir-v1.0.0-robust04.bge-base-en-v1.5',
    # 'signal': 'beir-v1.0.0-signal1m.bge-base-en-v1.5',
    # 'nq': 'beir-v1.0.0-nq.bge-base-en-v1.5',
    # 'hotpotqa': 'beir-v1.0.0-hotpotqa.bge-base-en-v1.5',
    # 'mrtydi-ar': 'mrtydi-v1.1-arabic',
    # 'mrtydi-bn': 'mrtydi-v1.1-bengali',
    # 'mrtydi-fi': 'mrtydi-v1.1-finnish',
    # 'mrtydi-id': 'mrtydi-v1.1-indonesian',
    # 'mrtydi-ja': 'mrtydi-v1.1-japanese',
    # 'mrtydi-ko': 'mrtydi-v1.1-korean',
    # 'mrtydi-ru': 'mrtydi-v1.1-russian',
    # 'mrtydi-sw': 'mrtydi-v1.1-swahili',
    # 'mrtydi-te': 'mrtydi-v1.1-telugu',
    # 'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'nq': 'beir-v1.0.0-nq-test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-test',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}
from run_utility_selection import utility_selection_pipeline, write_eval_file
from pyserini.search.lucene import LuceneHnswDenseSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from vllm import LLM, SamplingParams
from tqdm import tqdm
import transformers
import tempfile
import torch
import os
import json
import shutil
import argparse  # 添加argparse模块用于解析命令行参数
parser = argparse.ArgumentParser(description='Run retrieval and ranking with configurable top_k.')
parser.add_argument('--top_k', type=int, required=True, help='Number of top documents to retrieve and rerank')
parser.add_argument('--output_path', type=str, required=True, help='Number of top documents to retrieve and rerank')
parser.add_argument('--model_path', type=str, required=True, help='Number of top documents to retrieve and rerank')
args = parser.parse_args()
top_k = args.top_k  # 从命令行获取top_k值
def run_retriever(topics, searcher, lucene_bm25_searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(lucene_bm25_searcher.doc(hit.docid).raw())
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
            ranks.append({'query': query, 'query_id': qid, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(lucene_bm25_searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks

import os
if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        args.model_path,
        dtype=torch.float16,
        tensor_parallel_size=8
    )
    for data in ['hotpotqa']:
        print('#' * 20)
        print(f'Evaluation on {data}')
        print('#' * 20)
        path = args.output_path
        try:
            os.mkdir(path)
            print(f"目录创建成功")
        except FileExistsError:
            print(f"目录已经存在")
        temp_file = path+data+"_"+str(top_k)+".txt"
        if os.path.exists(temp_file):
            print(f"文件 {temp_file} 存在")
            continue  
        query_ids = []
        with open("nq_answer.jsonl", "r", encoding="utf-8") as file_r:
            for line in file_r:
                js = json.loads(line)
                query_ids.append(js["query_id"])
        # Retrieve passages using pyserini BM25.
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        faiss_searcher = LuceneHnswDenseSearcher.from_prebuilt_index(
            THE_INDEX[data],
            ef_search=1000,
            encoder='BgeBaseEn15')
        lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index(bm25_index[data])
        rank_results = run_retriever(topics, faiss_searcher, lucene_bm25_searcher, qrels, k=top_k)
        with open(temp_file, "w", encoding="utf-8") as file_w:
            for item in tqdm(rank_results):
                if data == 'nq':
                    if item['query_id'] in query_ids:
                        new_item = utility_selection_pipeline(llm, tokenizer, item, window_size=20)
                        file_w.write(json.dumps(new_item)+"\n")
                    else:
                        continue
                else:
                    new_item = utility_selection_pipeline(llm, tokenizer, item, window_size=20)
                    file_w.write(json.dumps(new_item)+"\n")
