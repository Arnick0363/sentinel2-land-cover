"""随机森林与高斯最大似然。两条路线使用同一套样本。"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class GaussianMLC:
    """逐类高斯最大似然。协方差加对角载荷，避免指数与原波段共线时矩阵奇异。"""

    def __init__(self, reg: float = 1e-3):
        self.reg = reg
        self.classes_: np.ndarray | None = None
        self._means: list[np.ndarray] = []
        self._inv: list[np.ndarray] = []
        self._logdet: list[float] = []
        self._log_prior: list[float] = []

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "GaussianMLC":
        labels = np.asarray(labels).astype(int)
        self.classes_ = np.unique(labels)
        self._means = []
        self._inv = []
        self._logdet = []
        self._log_prior = []
        for class_id in self.classes_:
            subset = features[labels == class_id]
            if len(subset) < 2:
                raise ValueError(f"类别 {class_id} 的训练样本少于 2，无法估计协方差")
            mean = subset.mean(axis=0)
            cov = np.cov(subset, rowvar=False)
            cov = np.atleast_2d(cov) + np.eye(subset.shape[1]) * self.reg
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError(f"类别 {class_id} 的协方差不可逆")
            self._means.append(mean)
            self._inv.append(np.linalg.inv(cov))
            self._logdet.append(float(logdet))
            self._log_prior.append(float(np.log(len(subset) / len(features))))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("最大似然模型尚未训练")
        scores = []
        for mean, inv, logdet, log_prior in zip(self._means, self._inv, self._logdet, self._log_prior):
            delta = features - mean
            quadratic = np.sum(delta @ inv * delta, axis=1)
            scores.append(-0.5 * quadratic - 0.5 * logdet + log_prior)
        best = np.argmax(np.vstack(scores), axis=0)
        return self.classes_[best]


def train_random_forest(features: np.ndarray, labels: np.ndarray, model_cfg: dict, seed: int) -> RandomForestClassifier:
    forest = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 100)),
        max_depth=model_cfg.get("max_depth", None),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        max_features=model_cfg.get("max_features", "sqrt"),
        class_weight=model_cfg.get("class_weight", "balanced_subsample"),
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(features, labels)
    return forest


def predict_raster(model, stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """对特征立方体预测。无效像元记为 0。"""
    height, width, n_features = stack.shape
    prediction = np.zeros((height, width), dtype=np.uint8)
    flat_valid = valid.reshape(-1)
    flat_features = stack.reshape(-1, n_features)
    if not flat_valid.any():
        return prediction
    usable = flat_valid & np.isfinite(flat_features).all(axis=1)
    predicted = np.zeros(flat_features.shape[0], dtype=np.uint8)
    if usable.any():
        predicted[usable] = model.predict(flat_features[usable]).astype(np.uint8)
    return predicted.reshape(height, width)


def majority_filter(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """3×3 众数滤波，去掉单像元椒盐。云掩膜内的像元保持 0。"""
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    counts = []
    for class_id in range(6):
        mask = (padded == class_id).astype(np.uint8)
        window = (
            mask[:-2, :-2]
            + mask[:-2, 1:-1]
            + mask[:-2, 2:]
            + mask[1:-1, :-2]
            + mask[1:-1, 1:-1]
            + mask[1:-1, 2:]
            + mask[2:, :-2]
            + mask[2:, 1:-1]
            + mask[2:, 2:]
        )
        counts.append(window)
    mode = np.argmax(np.stack(counts, axis=0), axis=0).astype(np.uint8)
    filtered = np.array(labels, copy=True)
    use = valid & (mode != 0)
    filtered[use] = mode[use]
    filtered[~valid] = 0
    return filtered
