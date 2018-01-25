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
from random import *

xrange = [-1, 1]
yrange = [-1, 1]

xs = []
ys = []

# Generate a randomly decided line (in vector form)
# vecline[0] is the slope, vecline[1] is the interception
p1 = [uniform(-1, 1), uniform(-1, 1)]
p2 = [uniform(-1, 1), uniform(-1, 1)]
plt.xlim(*xrange)
plt.ylim(*yrange)
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
    return 1 if np.dot(w_truth, [1,x,y]) > 0 else -1
    
    
# Generate n random points according to the target function
def gen_pts(n: int):
    while n > 0:
        n -= 1
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        yield [x, y, target(x, y)]
        
# plot the points
for i in gen_pts(100):
    # print(i)
    xs.append
    plt.plot(i[0], i[1], *['g+' if i[2]>0 else 'b_'])

plt.show()

# definition of the perceptron learning algorithm
def perceptron(Input, label):
    pass