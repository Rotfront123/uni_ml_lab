from knn_parzen import ParzenWindowClassifier
from model_selection import cross_val_score, train_test_split, LeaveOneOut
from scores import Accuracy
import numpy as np
import matplotlib.pyplot as plt

def plot_val_for_knn(scores, best_k):
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, 30)], scores, 'b-o', label="LOO ошибка")
    plt.plot(best_k, scores[best_k - 1], 'r*', label=f"Оптимальное k={best_k}")
    plt.xlabel('k (количество соседей)', fontsize=12)
    plt.ylabel('Ошибка LOO', fontsize=12)
    plt.title('Зависимость ошибки LOO от параметра k', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    #И без k=1
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(2, 30)], scores[1:], 'b-o', label="LOO ошибка")
    plt.plot(best_k, scores[best_k - 1], 'r*', label=f"Оптимальное k={best_k}")
    plt.xlabel('k (количество соседей)', fontsize=12)
    plt.ylabel('Ошибка LOO', fontsize=12)
    plt.title('Без k = 1', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
def compute_all_scores_for_knn(X, y, cv, scoring):
    scores = []
    for k in range(1, 30):
        clf = ParzenWindowClassifier(k=k)
        cv = LeaveOneOut()
        score = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        scores.append(score)
    return scores

def search_best_k(scores):
    return np.argmax(scores) + 1

def load_and_prepare_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data # массив numpy
    y = iris.target # массив numpy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    return X_train, y_train, X_test, y_test 


X_train, y_train, X_test, y_test = load_and_prepare_data()
scores = compute_all_scores_for_knn(X_train, y_train, cv=LeaveOneOut(), scoring=Accuracy())
print(scores)
best_k = search_best_k(scores)
plot_val_for_knn(scores, best_k)
clf = ParzenWindowClassifier(k=best_k)
y_pred = clf.fit(X_train, y_train).predict(X_test)
final_score = Accuracy()(y_test, y_pred)
print(f"Финальная оценка классификатора при оптимальном k={best_k}: {final_score}")
