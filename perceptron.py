#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:22:05 2018
Perceptron learing algorithm in 2d
As a presentation of the idea
This algorithm will be generalized in to suit more realistic scenarios.

@author: spencer
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd

xrange = [-1, 1]
yrange = [-1, 1]

xs = []
ys = []

# Generate a randomly decided line (in vector form)
# vecline[0] is the slope, vecline[1] is the interception
p1 = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
p2 = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
plt.xlim(*xrange)
plt.ylim(*yrange)
# plot the true line
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)


k = (p2[1] - p1[1])/(p2[0]-p1[0])
b = p1[1] - k * p1[0]

# tranform y = kx + b to Ax + By + C = 0
A = -k
B = 1
C = -b
w_truth = [C,A,B]

# Using the randomly generated parameter to define the real target function
def target(x, y):
    return 1 if np.dot(w_truth, np.transpose([1,x,y])) > 0 else -1
    

# Generate n random points according to the target function
def gen_pts(n: int):
    while n > 0:
        n -= 1
        x = rd.uniform(-1, 1)
        y = rd.uniform(-1, 1)
        yield [x, y, target(x, y)]
        
# plot the points
for i in gen_pts(100):
    # print(i)
    xs.append([i[0], i[1]])
    ys.append(i[2])
    plt.plot(i[0], i[1], *['g+' if i[2]>0 else 'b_'])


print(xs, ys)
# definition of the perceptron learning algorithm
def perceptron(Input, label, w):
    """
    Input: list of input value, [x1, x2, x3,...,xn]
    label: + or -
    w: initial weights for training
    """
    # print('w->', w)
    assert len(Input) == len(label)
    for i in range(len(Input)):
        x = [1]
        x.extend(Input[i])
        
        if np.dot(w, x) * label[i] < 0:
            w = list(np.add(w, np.multiply(x, label[i])))
    return w

# this train method doesn't work very good
def train(p, Input, label, w, iteration):
    if iteration == 0:
        return w
    else:
        # print('remaining iteration: ', iteration)
        return train(p, Input, label, p(Input, label, w), iteration-1)

w_hypoth = train(perceptron, xs, ys, [-4,2,1], 900)


print('w_truth -> ', w_truth)
print('w_hypoth -> ', w_hypoth)

# plot the true line
plt.plot([0, -w_hypoth[0]/w_hypoth[2]], [-w_hypoth[0]/w_hypoth[1], 0], 'yo-', linewidth=2)

plt.show()