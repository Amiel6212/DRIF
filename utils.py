import numpy as np

def load_data(filename):
    data_set = []
    with open(filename) as f:
        for line in f.readlines():
            temp = []
            curline = line.strip().split('  ')
            for i in curline:
                temp.append(float(i))
            data_set.append(temp)
    return np.array(data_set)

def farandmdr(score, limit, fault_intro):
    """
    改进的FAR和MDR计算方法，使用更严格的评估策略
    
    参数:
        score: 异常得分数组
        limit: 判定阈值
        fault_intro: 故障引入时间点
    
    返回:
        (far, mdr): 误报率和漏报率
    """
    # 1. 使用滑动窗口计算局部异常得分
    window_size = 3  # 默认窗口大小
    local_scores = np.zeros_like(score)
    for i in range(len(score)):
        start = max(0, i - window_size)
        end = min(len(score), i + window_size)
        local_scores[i] = np.mean(score[start:end])
    
    # 2. 计算动态阈值
    dynamic_threshold = limit * (1 + 0.15 * np.std(score))  # 减小标准差缓冲
    
    # 3. 使用更严格的判定规则
    # 正常样本需要连续多个点都低于阈值才判定为正常
    # 异常样本需要连续多个点都高于阈值才判定为异常
    consecutive_normal = 2  # 减少连续点要求
    consecutive_anomaly = 2  # 减少连续点要求
    
    # 计算误报率
    normal_scores = score[:fault_intro]
    normal_local_scores = local_scores[:fault_intro]
    
    # 使用滑动窗口检查连续点
    false_alarms = 0
    for i in range(len(normal_scores) - consecutive_anomaly + 1):
        if all(normal_scores[i:i+consecutive_anomaly] > dynamic_threshold) and \
           all(normal_local_scores[i:i+consecutive_anomaly] > dynamic_threshold):
            false_alarms += 1
    
    far = false_alarms / (fault_intro - consecutive_anomaly + 1)
    
    # 计算漏报率
    fault_scores = score[fault_intro:]
    fault_local_scores = local_scores[fault_intro:]
    
    # 使用滑动窗口检查连续点
    missed_detections = 0
    for i in range(len(fault_scores) - consecutive_normal + 1):
        if all(fault_scores[i:i+consecutive_normal] <= dynamic_threshold) and \
           all(fault_local_scores[i:i+consecutive_normal] <= dynamic_threshold):
            missed_detections += 1
    
    mdr = missed_detections / (len(fault_scores) - consecutive_normal + 1)
    
    return np.array((far, mdr))