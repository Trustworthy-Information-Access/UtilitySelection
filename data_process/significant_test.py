import json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, chi2_contingency
import argparse
import json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import argparse

def load_results(file_path):
    """加载结果文件并返回EM和F1列表"""
    em_scores = []
    f1_scores = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            em_scores.append(data['EM'])
            f1_scores.append(data['F1'])
    return np.array(em_scores), np.array(f1_scores)

def perform_significance_test(file1, file2):
    """执行显著性检验并打印结果"""
    # 加载数据
    em1, f1_1 = load_results(file1)
    em2, f1_2 = load_results(file2)
    
    # 检查样本大小是否一致
    if len(em1) != len(em2) or len(f1_1) != len(f1_2):
        print(len(em1))
        print(len(em2))
        raise ValueError("两个结果文件的样本数量不一致！")
    
    n = len(em1)
    print(f"样本数量: {n}")
    
    # 计算平均值和差异
    em_diff = em1 - em2
    f1_diff = f1_1 - f1_2
    
    # 执行配对t检验
    _, p_value_em_ttest = ttest_rel(em1, em2)
    _, p_value_f1_ttest = ttest_rel(f1_1, f1_2)
    
    # 执行Wilcoxon符号秩检验
    _, p_value_em_wilcox = wilcoxon(em_diff)
    _, p_value_f1_wilcox = wilcoxon(f1_diff)
    
    # 打印结果摘要
    print("\n===== 结果摘要 =====")
    print(f"{file1}平均EM: {np.mean(em1):.4f}, {file2}平均EM: {np.mean(em2):.4f}")
    print(f"{file1}平均F1: {np.mean(f1_1):.4f}, {file2}平均F1: {np.mean(f1_2):.4f}")
    print(f"EM平均差异: {np.mean(em_diff):.4f}, F1平均差异: {np.mean(f1_diff):.4f}")
    
    # 打印统计检验结果
    print("\n===== 统计显著性检验 =====")
    print(f"EM指标 (配对t检验) p值: {p_value_em_ttest:.6f}")
    print(f"EM指标 (Wilcoxon检验) p值: {p_value_em_wilcox:.6f}")
    print(f"F1指标 (配对t检验) p值: {p_value_f1_ttest:.6f}")
    print(f"F1指标 (Wilcoxon检验) p值: {p_value_f1_wilcox:.6f}")
    
    # 解释结果 (α=0.05)
    alpha = 0.05
    print("\n===== 结果解释 (α=0.05) =====")
    
    # 解释EM结果
    em_ttest_sig = "显著" if p_value_em_ttest < alpha else "不显著"
    em_wilcox_sig = "显著" if p_value_em_wilcox < alpha else "不显著"
    
    print(f"EM差异 (t检验): {em_ttest_sig} (p={p_value_em_ttest:.4f})")
    print(f"EM差异 (Wilcoxon): {em_wilcox_sig} (p={p_value_em_wilcox:.4f})")
    
    # 解释F1结果
    f1_ttest_sig = "显著" if p_value_f1_ttest < alpha else "不显著"
    f1_wilcox_sig = "显著" if p_value_f1_wilcox < alpha else "不显著"
    
    print(f"F1差异 (t检验): {f1_ttest_sig} (p={p_value_f1_ttest:.4f})")
    print(f"F1差异 (Wilcoxon): {f1_wilcox_sig} (p={p_value_f1_wilcox:.4f})")

    print("\n===== 推荐结论 =====")
    if p_value_em_ttest < alpha:
        print("在EM指标上，两个模型存在统计显著差异")
    else:
        print("在EM指标上，两个模型没有统计显著差异")
    
    if p_value_f1_ttest < alpha:
        print("在F1指标上，两个模型存在统计显著差异")
    else:
        print("在F1指标上，两个模型没有统计显著差异")
if __name__ == "__main__":
    file1 = "utility_selection/llama31_nq-answer_100_nw0_new_results.jsonl"
    for num in ["5", "10", "15", "20", "40", "60", "80", "100"]:
        file2 = "relevance_ranking/llama31_nq_filter-answer_"+num+"_nw0_results.jsonl"
        perform_significance_test(file1, file2)

   