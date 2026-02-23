import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator

class DistanceMetric(ABC):
    # Базовый класс для метрик расстояния. Позволяет вычислять как одиночные,
    # так и пакетные расстояния единообразно
    
    @abstractmethod  
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        pass
    
    def compute_batch(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        # Базовая реализация через цикл. Потомки могут переопределять
        # для векторизованных вычислений (особенно важно для больших данных)
        return np.array([self.compute(x, x_train) for x_train in X])
    

class EuclideanMetric(DistanceMetric):
    def compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((x - y) ** 2))
    
    def compute_batch(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        # Векторизованная версия: вычисляем расстояния до всех точек сразу
        # через broadcasting. Работает значительно быстрее базовой реализации
        return np.sqrt(np.sum((X - x) ** 2, axis=1))