Code for paper "Is Relevance Ranking Sufficient in RAG?"

This project aims to explore relevance ranking vs utility selection in RAG.  
Thanks for [RankGPT](https://github.com/sunnweiwei/RankGPT), [RankLLM](https://github.com/castorini/rank_llm) and [Utility Annotation for dense retrieval](https://github.com/Trustworthy-Information-Access/utility-focused-annotation). 

🎉🎉🎉 **[News]**: Checkpoints and training dataset of RankQwen and UtilityQwen in our paper are released on [UtilityQwen1.7B](https://huggingface.co/hengranZhang/UtilityQwen1.7B/tree/main)

# Quick example
## Installation
- Utility selection and relevance ranking needs [anserini](https://github.com/castorini/anserini) and [Pyserini](https://github.com/castorini/pyserini), which need Java. Please install Pyserini, and refer to the official documentation.
- Generation distillation needs [accelerate](https://github.com/huggingface/accelerate) and flash-attn.
## Datasets
100K training queries are sampled by [RankGPT](https://drive.google.com/file/d/1OgF4rj89FWSr7pl1c7Hu4x0oQYIMwhik/view?usp=share_link). Each query has the top 20 BM25-retrieved passages. 
The relevance ranking list and utility selection annotated by Qwen3 32B in the training set will be publicly linked after the anonymity period, because the files are too large. 
## Start 
### Relevance ranking and utility selection annotation
```
cd llm_utility
sh utility.sh 
```
### Generation distillation
```
cd rank_llm/training
sh run.sh 
```
### Relevance ranking and utility selection test 
```
cd RelevanceRank_UtilitySelection
sh run.sh 
```
Similar to the annotation list, all trained models' checkpoints in our work, e.g., RankQwen1.7B and UtilityQwen1.7B, will be publicly linked after the anonymity period.



