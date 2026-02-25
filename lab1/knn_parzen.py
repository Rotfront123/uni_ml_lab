import numpy as np
from typing import Optional, Tuple, Any, Iterator
from kernels import EpanechnikovKernel, Kernel
from metrics import DistanceMetric, EuclideanMetric

class ParzenWindowClassifier:
    def __init__(self, k: int = 5, metric: Optional[DistanceMetric] = None, kernel: Optional[Kernel] = None):
        self.k = k
        # Если метрика/ядро не заданы, используем стандартные
        self.metric = metric if metric else EuclideanMetric()
        self.kernel = kernel if kernel else EpanechnikovKernel()
        self.X_train = None
        self.y_train = None
        self.classes = None
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ParzenWindowClassifier':
        #fit Обучаем классификатор(загружаем данные)
        self.X_train = X
        self.y_train = y
        #уникальные классы для классификации
        self.classes = np.unique(y)
        #Возвращаем себя для возможности цепочки вызовов(like sklearn)
        return self
    def _compute_distance(self, x: np.ndarray) -> np.ndarray:
        #функция для нахождения расстояние от точки до множества точек
        return self.metric.compute_batch(x, self.X_train)
    # TODO: Можно оптимизировать через quickselect вместо полной сортировки (O(n) вместо O(n log n))
    def _get_k_distance(self, x: np.ndarray) -> float:
        #Ищем дистанцию до k+1-ого соседа
        distance = self._compute_distance(x)
        distance = np.sort(distance)
        return distance[self.k]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Возвращает numpy массив предсказаний
        # Если один объект, то оборачиваем в 2-мерный массив
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.zeros(len(X), dtype=self.y_train.dtype)
        for i, x in enumerate(X):
            predictions[i] = self._predict_one(x)
        
        return predictions
    def _predict_one(self, x: np.ndarray) -> Any:
        # Вычисляем ширину окна
        h = self._get_k_distance(x)
        if h == 0:
            weights = self.metric.compute_batch(x, self.X_train)
            weights = np.where(self.kernel.compute_batch(weights)==0, self.kernel.compute(0), 0)
        #Вычисляем веса(с учетом окна)
        else:
            weights = self.kernel.compute_batch(self.metric.compute_batch(x, self.X_train) / h)
        class_weights = {}
        for class_label in self.classes:
            # Создаем маску для объектов данного класса
            class_mask = (self.y_train == class_label)
            # Суммируем веса объектов этого класса
            class_weights[class_label] = np.sum(weights[class_mask])
        
        # Возвращаем класс с максимальным весом
        return max(class_weights, key=class_weights.get)