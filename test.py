#! /usr/bin/python2
# coding=utf-8
# author:chuzhuo

import numpy as np
import matplotlib.pyplot as plt

x_axis = range(0,9,1)
y_axis = range(0,9,1)
y_axis_2 = range(0,9,1)

for i in range(0,9,1):
    y_axis[i] = x_axis[i]
    y_axis_2[i] = x_axis[i] * x_axis[i]

fig = plt.figure(figsize=(7.00, 5.25), clear=True)
axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
axe_1.plot(x_axis,y_axis,linewidth=0.5, color="black",marker=".", label="y=2x")
axe_1.plot(x_axis,y_axis_2,linewidth=0.5, color="black",marker="*", label="y=x*x")
axe_1.legend()
axe_1.tick_params(top='on',right='on',direction='in')
axe_1.annotate("y=2*x",xy=(1,2),xytext=(3,50),arrowprops=dict(
    arrowstyle='->'
))
fig.show(warn=False)