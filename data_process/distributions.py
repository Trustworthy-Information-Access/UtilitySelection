import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
# from scipy.stats import gaussian_kde
data_type = "nq"
nums_nq = []
with open("results/annotation_utility_answer_both/results_first10_merge_remian/"+data_type+".jsonl", "r", encoding="utf-8") as file_r:
    for line in file_r:
        js = json.loads(line)
        nums_nq.append(len(js["hits"]))
data_type = "hotpotqa"
nums_hotpotqa = []
with open("results/annotation_utility_answer_both/results_first10_merge_remian/"+data_type+".jsonl", "r", encoding="utf-8") as file_r:
    for line in file_r:
        js = json.loads(line)
        nums_hotpotqa.append(len(js["hits"]))


# 数据集标签
dataset_labels = ["NQ", "HotpotQA"]
datasets = [nums_nq, nums_hotpotqa]

# 创建1行3列的子图
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 设置全局标题
fig.suptitle('Numbers of Selected Documents Distribution on Three datasets', fontsize=20, y=1.02)

# 统一颜色方案
colors = ['#1f77b4', '#ff7f0e']
mean_line_colors = ['#d62728', '#9467bd']

# 遍历三个数据集
for i, (data, label, color, mean_color) in enumerate(zip(datasets, dataset_labels, colors, mean_line_colors)):
    ax = axes[i]
    
    # 计算统计量
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    
    # 绘制直方图
    n, bins, patches = ax.hist(data, bins=30, density=True, 
                              color=color, alpha=0.6, 
                              edgecolor='white')
    

    
    # 添加平均值线
    ax.axvline(mean_val, color=mean_color, linestyle='--', linewidth=3, label=f"Average Numbers: {mean_val:.2f}")
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    # 添加标题和标签
    # ax.set_title(f'{label} (n={len(data)})', fontsize=20)
    ax.set_xlabel(f'{label} (n={len(data)})', fontsize=20)
    
    # 只有第一列子图显示y轴标签
    if i == 0:
        ax.set_ylabel('Distribution', fontsize=20)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    ax.legend(fontsize=15)
    

# 调整子图间距
plt.tight_layout()

# 导出为PDF文件
output_filename = "images/Distribution.pdf"
plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
print(f"saved: {output_filename}")

# 显示图表
plt.show()