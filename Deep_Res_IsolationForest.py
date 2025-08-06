import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from utils import farandmdr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class ResidualBlock:
    def __init__(self, input_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler1 = RobustScaler(quantile_range=(1, 99))
        self.scaler2 = RobustScaler(quantile_range=(1, 99))
        self.pca1 = PCA(n_components=0.999)
        self.pca2 = PCA(n_components=0.999)
        self.is_fitted = False
        
    def fit(self, X):
        # 添加更多统计特征
        X_stats = np.concatenate([
            X,
            np.mean(X, axis=1, keepdims=True),
            np.std(X, axis=1, keepdims=True),
            np.max(X, axis=1, keepdims=True),
            np.min(X, axis=1, keepdims=True),
            np.median(X, axis=1, keepdims=True),
            np.percentile(X, 25, axis=1, keepdims=True),
            np.percentile(X, 75, axis=1, keepdims=True),
            # 添加差分特征
            np.diff(X, axis=0, prepend=X[0:1]),
            # 添加滑动窗口统计特征
            np.convolve(X.mean(axis=1), np.ones(5)/5, mode='same').reshape(-1, 1),
            np.convolve(X.std(axis=1), np.ones(5)/5, mode='same').reshape(-1, 1),
            # 添加新的统计特征
            np.var(X, axis=1, keepdims=True),
            stats.skew(X, axis=1, keepdims=True),
            stats.kurtosis(X, axis=1, keepdims=True),
            # 添加趋势特征
            np.gradient(X, axis=0),
            np.gradient(np.gradient(X, axis=0), axis=0)
        ], axis=1)
        
        # 第一层变换
        X_scaled1 = self.scaler1.fit_transform(X_stats)
        X_pca1 = self.pca1.fit_transform(X_scaled1)
        
        # 计算残差
        X_reconstructed = self.scaler1.inverse_transform(self.pca1.inverse_transform(X_pca1))
        residual = X_stats - X_reconstructed
        
        # 第二层变换
        X_scaled2 = self.scaler2.fit_transform(residual)
        self.pca2.fit(X_scaled2)
        
        self.is_fitted = True
        self.n_features = X.shape[1]
        
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("ResidualBlock must be fitted before transform")
        
        # 添加更多统计特征
        X_stats = np.concatenate([
            X,
            np.mean(X, axis=1, keepdims=True),
            np.std(X, axis=1, keepdims=True),
            np.max(X, axis=1, keepdims=True),
            np.min(X, axis=1, keepdims=True),
            np.median(X, axis=1, keepdims=True),
            np.percentile(X, 25, axis=1, keepdims=True),
            np.percentile(X, 75, axis=1, keepdims=True),
            # 添加差分特征
            np.diff(X, axis=0, prepend=X[0:1]),
            # 添加滑动窗口统计特征
            np.convolve(X.mean(axis=1), np.ones(5)/5, mode='same').reshape(-1, 1),
            np.convolve(X.std(axis=1), np.ones(5)/5, mode='same').reshape(-1, 1),
            # 添加新的统计特征
            np.var(X, axis=1, keepdims=True),
            stats.skew(X, axis=1, keepdims=True),
            stats.kurtosis(X, axis=1, keepdims=True),
            # 添加趋势特征
            np.gradient(X, axis=0),
            np.gradient(np.gradient(X, axis=0), axis=0)
        ], axis=1)
            
        # 第一层变换
        X_scaled1 = self.scaler1.transform(X_stats)
        X_pca1 = self.pca1.transform(X_scaled1)
        
        # 计算残差
        X_reconstructed = self.scaler1.inverse_transform(self.pca1.inverse_transform(X_pca1))
        residual = X_stats - X_reconstructed
        
        # 第二层变换
        X_scaled2 = self.scaler2.transform(residual)
        X_pca2 = self.pca2.transform(X_scaled2)
        
        return X_pca1, X_pca2, residual

class DRIF:
    def __init__(self, n_layers=2, n_estimators=200, contamination=0.0001, 
                 original_ratio=0.3, confi_limit=0.99, fault_intro=160):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.original_ratio = original_ratio
        self.confi_limit = confi_limit
        self.fault_intro = fault_intro
        
        # 初始化模型组件
        self.residual_blocks = []
        self.if_models = []
        self.priors = []
        
    def _create_isolation_forest(self):
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            max_samples='auto',
            bootstrap=True,
            n_jobs=-1
        )
    
    def fit(self, X):
        """训练模型"""
        current_data = X.copy()
        
        for layer in range(self.n_layers):
            print(f"\nTraining layer {layer + 1}/{self.n_layers}...")
            
            # 创建并拟合残差块
            residual_block = ResidualBlock(current_data.shape[1])
            residual_block.fit(current_data)
            self.residual_blocks.append(residual_block)
            
            # 特征变换
            X_pca1, X_pca2, residual = residual_block.transform(current_data)
            
            # 训练两个Isolation Forest模型
            if_model1 = self._create_isolation_forest()
            if_model2 = self._create_isolation_forest()
            
            if_model1.fit(X_pca1)
            if_model2.fit(X_pca2)
            
            self.if_models.append((if_model1, if_model2))
            
            # 计算先验概率
            scores1 = -if_model1.score_samples(X_pca1)
            scores2 = -if_model2.score_samples(X_pca2)
            
            # 计算自适应权重
            weight1 = np.exp(-np.std(scores1))
            weight2 = np.exp(-np.std(scores2))
            total_weight = weight1 + weight2
            weight1 /= total_weight
            weight2 /= total_weight
            
            # 保存统计信息
            self.priors.append({
                'mean1': np.mean(scores1),
                'std1': np.std(scores1),
                'mean2': np.mean(scores2),
                'std2': np.std(scores2),
                'weight1': weight1,
                'weight2': weight2,
                'min1': np.min(scores1),
                'max1': np.max(scores1),
                'min2': np.min(scores2),
                'max2': np.max(scores2)
            })
            
            # 更新输入数据
            current_data = residual
    
    def transform(self, X):
        """异常检测"""
        all_scores = []
        current_data = X.copy()
        
        for layer in range(self.n_layers):
            # 获取当前层的组件
            residual_block = self.residual_blocks[layer]
            if_model1, if_model2 = self.if_models[layer]
            prior_stats = self.priors[layer]
            
            # 特征变换
            X_pca1, X_pca2, residual = residual_block.transform(current_data)
            
            # 计算异常分数
            scores1 = -if_model1.score_samples(X_pca1)
            scores2 = -if_model2.score_samples(X_pca2)
            
            # 使用自适应归一化
            scores1_norm = (scores1 - prior_stats['min1']) / (prior_stats['max1'] - prior_stats['min1'])
            scores2_norm = (scores2 - prior_stats['min2']) / (prior_stats['max2'] - prior_stats['min2'])
            
            # 使用保存的自适应权重
            weight1 = prior_stats['weight1']
            weight2 = prior_stats['weight2']
            
            # 计算后验概率，使用更敏感的sigmoid函数
            posterior1 = scores1_norm * (1 / (1 + np.exp(-(scores1_norm - prior_stats['mean1']) / (prior_stats['std1'] * 0.1))))
            posterior2 = scores2_norm * (1 / (1 + np.exp(-(scores2_norm - prior_stats['mean2']) / (prior_stats['std2'] * 0.1))))
            
            # 使用自适应权重组合
            layer_scores = weight1 * posterior1 + weight2 * posterior2
            
            # 使用动态阈值
            dynamic_percentile = np.clip(100 - (layer + 1) * 1.5, 98, 99.5)
            layer_scores = (layer_scores - np.percentile(layer_scores, 0.01)) / (np.percentile(layer_scores, dynamic_percentile) - np.percentile(layer_scores, 0.01))
            
            all_scores.append(layer_scores)
            
            # 更新输入数据
            current_data = residual
        
        return np.stack(all_scores)
    
    def decision_function(self, X):
        """决策函数"""
        scores = self.transform(X)
        # 使用两层分数的加权平均，进一步增加第一层的权重
        return 0.95 * scores[0] + 0.05 * scores[1]

    def detect(self, X, limit, fault_intro):
        """检测函数"""
        scores = self.transform(X)
        result = np.zeros((self.n_layers, 2))
        
        for layer in range(self.n_layers):
            result[layer] = farandmdr(scores[layer], limit, fault_intro)
            print(f"\nLayer {layer + 1} Results:")
            print(f"AUC = {result[layer][0]:.4f}")
            print(f"FAR = {result[layer][1]:.2f}%")
            print(f"MDR = {result[layer][2]:.2f}%")
        
        # 计算加权平均指标，进一步增加第一层的权重
        weighted_auc = 0.95 * result[0][0] + 0.05 * result[1][0]
        weighted_far = 0.95 * result[0][1] + 0.05 * result[1][1]
        weighted_mdr = 0.95 * result[0][2] + 0.05 * result[1][2]
        
        print("\nOverall Weighted Metrics:")
        print(f"Weighted AUC = {weighted_auc:.4f}")
        print(f"Weighted FAR = {weighted_far:.2f}%")
        print(f"Weighted MDR = {weighted_mdr:.2f}%")
        
        return scores, result