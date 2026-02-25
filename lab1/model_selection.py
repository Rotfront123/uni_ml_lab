import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator

class LeaveOneOut:
    # Реализация LOO кросс-валидации - каждый объект по очереди становится тестовым
    
    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Генерирует индексы для train/test при LOO
        n_samples = len(X)
        for i in range(n_samples):
            # Все индексы кроме i - в train, i - в test
            train_ind = np.concatenate([np.arange(i), np.arange(i+1, n_samples)])
            test_ind = np.array([i])
            yield train_ind, test_ind

def cross_val_score(estimator, X: np.ndarray, y: np.ndarray, cv, scoring) -> float:
    # Вычисляет среднюю метрику качества на кросс-валидации
    # estimator, cv, scoring не аннотируем типами, т.к. могут быть разные реализации
    sum_scores = 0
    for train_ind, test_ind in cv.split(X):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        score = scoring(y_pred, y_test)
        sum_scores += score
    return sum_scores / len(X)

def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.25, 
    random_state: Optional[int] = 42, 
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Разбивает данные на обучающую и тестовую выборки
    
    n_test = int(len(X) * test_size)
    indices = np.arange(len(X))
    
    if shuffle:
        np.random.seed(random_state)
        # shuffle изменяет массив на месте(перемешивает)
        np.random.shuffle(indices)
    
    # Индексы для train/test
    ind_train = indices[n_test:]
    ind_test = indices[:n_test]
    
    X_train = X[ind_train]
    X_test = X[ind_test]
    y_train = y[ind_train]
    y_test = y[ind_test]

    return X_train, X_test, y_train, y_test