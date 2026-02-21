import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # массив numpy
y = iris.target # массив numpy

class Kernel(ABC):
    #Вычисление ядра для одного объекта
    @abstractmethod  
    def compute(self, r: float) -> float:
        pass
    #Делаем объект вызываемым(callable) Пока вопрос
    def __call__(self, r: float) -> float:
        return self.compute(r)
    #Вычисление ядра для множества объектов
    def compute_batch(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.compute(elem) for elem in x])

#Абстрактный класс для создания кастомных метрик
class DistanceMetric(ABC):
    #Вычисление расстояние от точки до точки
    @abstractmethod  
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        pass
    #Вычисление расстояние от точки до множества точек
    def compute_batch(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.array([self.compute(x, x_train) for x_train in X])

#Ядро Епанечкова
class EpanechnikovKernel(Kernel):
    def compute(self, r: float) -> float:
        if abs(r) <= 1:
            return (3.0 / 4.0) * (1.0 - r ** 2)
        return 0.0
    def compute_batch(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) <= 1, (3.0 / 4.0) * (1.0 - x ** 2), 0)
        
        

#Евклидово расстояние
class EuclideanMetric(DistanceMetric):
    def compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((x - y) ** 2))
    def compute_batch(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((X - x) ** 2, axis=1))


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class Accuracy(BaseMetric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

class LeaveOneOut:
    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        for i in range(n_samples):
            train_ind = np.concatenate(np.arange(i), np.arange(i+1, n_samples))
            test_ind = np.array([i])
            yield train_ind, test_ind

def cross_val_score(estimator, cv, scoring, X: np.ndarray, y: np.ndarray) -> float:
    sum_scores = 0
    for train_ind, test_ind in cv.split(X):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        score = scoring(y_pred, y_test)
        sum_scores += score
    return sum_scores/len(X)

#Классификатор методом k ближйших соседей, парзеновское окно
class ParzenWindowClassifier:
    def __init__(self, k: int = 5, metric: Optional[DistanceMetric] = None, kernel: Optional[Kernel] = None):
        self.k = k
        self.metric = metric if metric else EuclideanMetric()
        self.kernel = kernel if kernel else EpanechnikovKernel()
        self.X_train = None
        self.Y_train = None
        self.classes = None
    #fit Обучаем классификатор(загружаем данные)
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'ParzenWindowClassifier':
        self.X_train = X
        self.Y_train = Y
        #уникальные классы для классификации
        self.classes = np.unique(Y)
        #Возвращаем себя для возможности цепочки вызовов
        return self
    #функция для нахождения расстояние от точки до множества точек
    def _compute_distance(self, x: np.ndarray) -> np.ndarray:
        return self.metric.compute_batch(x, self.X_train)
    #Ищем дистанцию до k-ого соседа
    #TODO: Можно делать не за nlonn а за n
    def _get_k_distance(self, x: np.ndarray) -> int:
        distance = self._compute_distance(x)
        distance = np.sort(distance)
        return distance[self.k-1]
    
    #TODO Дописать случай h=0
    def predict(self, x: np.ndarray) -> int:

        #Дистанция до k-ого объекта
        h = self._get_k_distance(x)
        #Вычисляем веса(с учетом окна)
        weights = self.kernel.compute_batch(self.metric.compute_batch(x, self.X_train) / h)

def search_Kmeans_grid(X, y, cv, scoring):
    scores = []
    for k in range(1, len(X)):
        score = cross_val_score(ParzenWindowClassifier(k=k), X, y, cv=cv, scoring=scoring)
        scores.append(score)
    return scores

#TODO написать функцию построения графиков
def plot_val_result(X, y, cv, scoring):
    scores = search_Kmeans_grid(X, y, cv=cv, scoring=scoring)
    
