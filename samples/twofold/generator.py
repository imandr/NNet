import math, random, numpy as np

w = 0.3

def point(r, w):
    r += w*(random.random()*2-1)
    phi = random.random()*2*math.pi
    return np.array([r*math.cos(phi), r*math.sin(phi)])

def generate(n):
    x = []
    y = []
    for _ in range(n):
        r, i = (0.0, 0) if random.random() < 0.5 else (1.0, 1)
        xi, yi = point(r, w), [0.0, 0.0]
        yi[i] = 1.0
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)
