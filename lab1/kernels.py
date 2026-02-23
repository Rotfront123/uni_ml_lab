import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator

class Kernel(ABC):
    # Базовый класс для всех ядер. Реализует паттерн "Шаблонный метод"
    # через абстрактный compute(), который должны переопределять потомки
    
    @abstractmethod  
    def compute(self, r: float) -> float:
        pass
    
    def __call__(self, r: float) -> float:
        # Позволяет использовать объект ядра как функцию: kernel(r)
        return self.compute(r)
    
    def compute_batch(self, x: np.ndarray) -> np.ndarray:
        # Базовая реализация для пакетной обработки.
        # Потомки могут переопределять для оптимизации (векторизации)
        return np.array([self.compute(elem) for elem in x])
    

class EpanechnikovKernel(Kernel):
    def compute(self, r: float) -> float:
        # Ядро Епанечкова: (3/4)(1 - r²) при |r| ≤ 1, иначе 0
        # Оптимально по среднеквадратичной ошибке среди всех ядер
        if abs(r) <= 1:
            return (3.0 / 4.0) * (1.0 - r ** 2)
        return 0.0
    
    def compute_batch(self, x: np.ndarray) -> np.ndarray:
        # Переопределяем для векторизованной обработки (быстрее чем в базовом классе)
        return np.where(np.abs(x) <= 1, (3.0 / 4.0) * (1.0 - x ** 2), 0)