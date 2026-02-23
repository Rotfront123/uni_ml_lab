#Классификатор методом k ближйших соседей, парзеновское окно
import numpy as np
from typing import Optional
from metrics import EuclideanMetric, DistanceMetric
from kernels import EpanechnikovKernel, Kernel
class ParzenWindowClassifier:
    def __init__(self, k: int = 5, metric: Optional[DistanceMetric] = None, kernel: Optional[Kernel] = None):
        self.k = k
        # Если метрика/ядро не заданы, используем стандартные
        self.metric = metric if metric else EuclideanMetric()
        self.kernel = kernel if kernel else EpanechnikovKernel()
        self.X_train = None
        self.Y_train = None
        self.classes = None
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'ParzenWindowClassifier':
        #fit Обучаем классификатор(загружаем данные)
        self.X_train = X
        self.Y_train = Y
        #уникальные классы для классификации
        self.classes = np.unique(Y)
        #Возвращаем себя для возможности цепочки вызовов(like sklearn)
        return self
    def _compute_distance(self, x: np.ndarray) -> np.ndarray:
        #функция для нахождения расстояние от точки до множества точек
        return self.metric.compute_batch(x, self.X_train)
    # TODO: Можно оптимизировать через quickselect вместо полной сортировки (O(n) вместо O(n log n))
    def _get_k_distance(self, x: np.ndarray) -> int:
        #Ищем дистанцию до k-ого соседа
        distance = self._compute_distance(x)
        distance = np.sort(distance)
        return distance[self.k-1]
    
    # TODO: Обработать случай h=0 (когда k-й сосед на нулевом расстоянии)
    def predict(self, x: np.ndarray) -> int:
        # Вычисляем ширину окна
        h = self._get_k_distance(x)
        #Вычисляем веса(с учетом окна)
        weights = self.kernel.compute_batch(self.metric.compute_batch(x, self.X_train) / h)
        class_weights = {}
        for class_label in self.classes:
            # Создаем маску для объектов данного класса
            class_mask = (self.y_train == class_label)
            # Суммируем веса объектов этого класса
            class_weights[class_label] = np.sum(weights[class_mask])
        
        # Возвращаем класс с максимальным весом
        return max(class_weights, key=class_weights.get)
        
