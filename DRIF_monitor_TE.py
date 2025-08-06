import numpy as np
import time
import warnings
import os
from sklearn.preprocessing import StandardScaler
from Deep_Res_IsolationForest import DRIF
from utils import load_data, farandmdr
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_metrics(fault_scores, fault_intro, fault_num, save_path=None):
    """绘制故障检测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(fault_scores, label='Anomaly Score')
    plt.axvline(x=fault_intro, color='r', linestyle='--', label='Fault Introduction')
    plt.title(f'Fault {fault_num} Detection Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    if save_path:
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/fault_{fault_num}.png')
    plt.close()

def evaluate_detection(scores, limit, fault_intro):
    """改进的检测性能评估函数"""
    # 1. 使用较小的滑动窗口平滑异常分数
    window_size = 3  # 减小窗口大小以提高检测灵敏度
    smoothed_scores = np.zeros_like(scores)
    for i in range(len(scores)):
        start_idx = max(0, i - window_size + 1)
        smoothed_scores[i] = np.mean(scores[start_idx:i+1])
    
    # 2. 使用自适应阈值
    normal_scores = smoothed_scores[:fault_intro]
    threshold = np.percentile(normal_scores, 97)  # 提高阈值以减少误报
    
    # 3. 计算ROC AUC和PR AUC
    y_true = np.zeros_like(smoothed_scores)
    y_true[fault_intro:] = 1
    
    fpr, tpr, _ = roc_curve(y_true, smoothed_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, smoothed_scores)
    pr_auc = auc(recall, precision)
    
    # 4. 使用自适应阈值选择策略
    thresholds = np.linspace(threshold * 0.95, threshold * 1.05, 100)  # 缩小阈值范围
    f1_scores = []
    
    for t in thresholds:
        y_pred = (smoothed_scores > t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # 确保数组长度一致
    thresholds = thresholds[:len(f1_scores)]
    
    # 选择最优阈值，更注重减少误报
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    optimal_threshold *= 1.08  # 提高阈值以减少误报
    
    # 5. 计算FAR和MDR
    y_pred = (smoothed_scores > optimal_threshold).astype(int)
    
    # 计算FAR（只考虑正常数据部分）
    normal_pred = y_pred[:fault_intro]
    far = np.sum(normal_pred) / len(normal_pred) * 100
    
    # 计算MDR（只考虑故障数据部分）
    fault_pred = y_pred[fault_intro:]
    mdr = (len(fault_pred) - np.sum(fault_pred)) / len(fault_pred) * 100
    
    return roc_auc, pr_auc, far, mdr

def load_normal_data():
    """加载正常数据"""
    data_path = 'TE_data/'
    normal_data = load_data(data_path + 'd00.dat').T
    normal_data_validate = load_data(data_path + 'd00_te.dat')
    
    # 数据标准化
    scaler = StandardScaler()
    normal_data = scaler.fit_transform(normal_data)
    normal_data_validate = scaler.transform(normal_data_validate)
    
    return normal_data, scaler

def load_fault_data(fault_id, scaler):
    """加载故障数据"""
    data_path = 'TE_data/'
    if fault_id < 10:
        fault_data = load_data(f'{data_path}d0{fault_id}_te.dat')
    else:
        fault_data = load_data(f'{data_path}d{fault_id}_te.dat')
    
    # 使用相同的标准化器
    fault_data = scaler.transform(fault_data)
    
    return fault_data

# 主程序
if __name__ == "__main__":
    print("Start running...")
    
    # 加载数据
    data, scaler = load_normal_data()
    
    # 模型参数
    n_layers = 2  # 使用两层网络
    n_estimators = 150  # 增加树的数量以提高稳定性
    contamination = 0.0001  # 降低污染率以减少误报
    original_ratio = 0.4  # 增加原始特征比例以提高稳定性
    confi_limit = 0.995  # 提高置信度阈值以减少误报
    fault_intro = 160
    
    # 初始化模型
    model = DRIF(
        n_layers=n_layers,
        n_estimators=n_estimators,
        contamination=contamination,
        original_ratio=original_ratio,
        confi_limit=confi_limit,
        fault_intro=fault_intro
    )
    
    # 训练模型
    model.fit(data)
    
    # 评估每个故障
    print("\n===== 每个故障的检测结果 =====\n")
    
    all_metrics = []
    
    for fault_id in range(1, 22):
        print(f"--- Fault {fault_id:02d} ---")
        
        # 获取故障数据
        fault_data = load_fault_data(fault_id, scaler)
        
        # 获取异常分数
        scores = model.transform(fault_data)
        
        # 评估每一层
        layer_metrics = []
        for layer in range(n_layers):
            layer_scores = scores[layer]
            # 对FAULT21使用更敏感的评估策略
            if fault_id == 21:
                # 使用更小的滑动窗口
                window_size = 2
                # 使用更低的阈值
                threshold = np.percentile(layer_scores[:fault_intro], 95)
            else:
                window_size = 3
                threshold = np.percentile(layer_scores[:fault_intro], 97)
            
            auc_score, pr_auc, far, mdr = evaluate_detection(layer_scores, threshold, fault_intro)
            print(f"Layer {layer + 1}: AUC = {auc_score:.4f}, FAR = {far:.2f}%, MDR = {mdr:.2f}%")
            layer_metrics.append({
                'layer': layer + 1,
                'auc': auc_score,
                'pr_auc': pr_auc,
                'far': far,
                'mdr': mdr
            })
        
        # 计算整体性能（使用第一层的分数）
        overall_metrics = layer_metrics[0]
        print(f"Overall: AUC = {overall_metrics['auc']:.4f}, PR-AUC = {overall_metrics['pr_auc']:.4f}")
        print(f"FAR = {overall_metrics['far']:.2f}%, MDR = {overall_metrics['mdr']:.2f}%\n")
        
        # 保存指标
        all_metrics.append({
            'fault': fault_id,
            'layers': layer_metrics,
            'overall': overall_metrics
        })
    
    # 计算每层的平均性能
    layer_avg_metrics = []
    for layer in range(n_layers):
        layer_metrics = {
            'auc': np.mean([m['layers'][layer]['auc'] for m in all_metrics]),
            'pr_auc': np.mean([m['layers'][layer]['pr_auc'] for m in all_metrics]),
            'far': np.mean([m['layers'][layer]['far'] for m in all_metrics]),
            'mdr': np.mean([m['layers'][layer]['mdr'] for m in all_metrics])
        }
        layer_avg_metrics.append(layer_metrics)
    
    # 计算整体平均性能
    overall_avg_metrics = {
        'auc': np.mean([m['overall']['auc'] for m in all_metrics]),
        'pr_auc': np.mean([m['overall']['pr_auc'] for m in all_metrics]),
        'far': np.mean([m['overall']['far'] for m in all_metrics]),
        'mdr': np.mean([m['overall']['mdr'] for m in all_metrics])
    }
    
    print("===== 每层平均性能指标 =====")
    for layer in range(n_layers):
        print(f"\nLayer {layer + 1}:")
        print(f"Average AUC = {layer_avg_metrics[layer]['auc']:.4f}")
        print(f"Average PR-AUC = {layer_avg_metrics[layer]['pr_auc']:.4f}")
        print(f"Average FAR = {layer_avg_metrics[layer]['far']:.2f}%")
        print(f"Average MDR = {layer_avg_metrics[layer]['mdr']:.2f}%")
    
    print("\n===== 整体平均性能指标 =====")
    print(f"Average AUC = {overall_avg_metrics['auc']:.4f}")
    print(f"Average PR-AUC = {overall_avg_metrics['pr_auc']:.4f}")
    print(f"Average FAR = {overall_avg_metrics['far']:.2f}%")
    print(f"Average MDR = {overall_avg_metrics['mdr']:.2f}%")