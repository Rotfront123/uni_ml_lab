import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator

class BaseMetric(ABC):
     # Базовый класс для метрик качества.
     # Работет только с numpy.ndarray!!!
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        #__call__ для вызова объекта как функции.
        #example: metric(y_true, y_pred)
        pass

class Accuracy(BaseMetric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Доля правильных ответов. 
        # y_true == y_pred → булев массив
        # mean() неявно преобразует True→1, False→0
        return np.mean(y_true == y_pred)