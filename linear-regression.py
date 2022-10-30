import numpy as np
import random


def predict(w, x):
    return x @ w

def mse(w, x, y):
    return ((predict(w, x) - y)**2).mean()

def gradient_descent(x, y, lr, epochs):
    x = np.c_[np.array(x), np.ones(len(x))]
    y = np.array(y)
    errors = []
    w = np.array([random.random() for _ in range(2)])

    for e in range(epochs):
        #print(w, x, predict(w, x))
        grad = 2 * np.dot((predict(w, x) - y), x) / len(x)
        w -= grad * lr

        errors.append(mse(w, x, y))
        print(f'{e:>5}: Loss={errors[-1]}', end='\n')
        
    return w

if __name__ == '__main__':

    true_a = random.uniform(0, 1)
    true_b = random.uniform(0, 1)
    f = lambda x: true_a * x + true_b
    g = lambda x: f(x) + random.gauss(0, 0.02)

    n = 100

    X = [random.random() for _ in range(n)]
    Y = list(map(f, X))

    model = gradient_descent(X, Y, 0.05, 1000)
    
    print(f'{[true_a, true_b]} - {model} = {[true_a, true_b] - model}')
