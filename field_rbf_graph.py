

import sys
import numpy as np
import string
import locale
from locale import atof

#from __future__ import division, print_function, absolute_import
from numpy.testing import (assert_, assert_array_almost_equal,
                           assert_almost_equal, run_module_suite)
from numpy import linspace, sin, random, exp, allclose
from scipy.interpolate.rbf import Rbf

import matplotlib.pyplot as plt
from matplotlib import cm

#combination constraints of 2D implicit fields

#combined field
x, y, z = [], [],[]
with open("field.txt") as C:
    for eachline in C:
        tmp = eachline.split()
        x.append(atof(tmp[0]))
        y.append(atof(tmp[1]))
        z.append(atof(tmp[4]))
print (x,y,z)

#field A
x1, y1, z1 = [], [],[]
with open("field1.txt") as A:
    for eachline in A:
        tmp1 = eachline.split()
        x1.append(atof(tmp1[0]))
        y1.append(atof(tmp1[1]))
        z1.append(atof(tmp1[4]))

#field B
x2, y2, z2 = [], [],[]
with open("field2.txt") as B:
    for eachline in B:
        tmp2 = eachline.split()
        x2.append(atof(tmp2[0]))
        y2.append(atof(tmp2[1]))
        z2.append(atof(tmp2[4]))

#minimum bounding box
minX = min(x)
minY = min(y)
maxX = max(x)
maxY = max(y)
#scale bounding box
percent = 0.1
minX = minX - (maxX-minX) * percent
maxX = maxX + (maxX-minX) * percent
minY = minY - (maxY-minY) * percent
maxY = maxY + (maxY-minY) * percent

#FUNCTIONS = ('multiquadric', 'inverse multiquadric', 'gaussian', 'cubic', 'quintic', 'thin-plate', 'linear')

# 2-d tests - setup scattered data
tiX = np.linspace(minX, maxX, 500)
tiY = np.linspace(minY, maxY, 500)

XI, YI = np.meshgrid(tiX, tiY)

# use RBF
rbf1 = Rbf(x1, y1, z1, epsilon=2)
rbf2 = Rbf(x2, y2, z2, epsilon=2)
ZI1 = rbf1(XI, YI)
ZI2 = rbf2(XI, YI)

#Union of A and B
ZI5 = np.array(ZI1)
for i in range(0,XI.shape[0]):
    for j in range(0,YI.shape[0]):
        if ZI1[i][j] > ZI2[i][j]:
            ZI5[i][j] = ZI2[i][j]
        else:
            ZI5[i][j] = ZI1[i][j]

#Intersection of A and B
ZI6 = np.array(ZI1)
for i in range(0,XI.shape[0]):
    for j in range(0,YI.shape[0]):
        if ZI1[i][j] > ZI2[i][j]:
            ZI6[i][j] = ZI1[i][j]
        else:
            ZI6[i][j] = ZI2[i][j]

#A - B
ZI7 = np.array(ZI1)
for i in range(0,XI.shape[0]):
    for j in range(0,YI.shape[0]):
        if ZI1[i][j] > -ZI2[i][j]:
            ZI7[i][j] = ZI1[i][j]
        else:
            ZI7[i][j] = -ZI2[i][j]

#B - A
ZI8 = np.array(ZI1)
for i in range(0,XI.shape[0]):
    for j in range(0,YI.shape[0]):
        if ZI2[i][j] > -ZI1[i][j]:
            ZI8[i][j] = ZI2[i][j]
        else:
            ZI8[i][j] = -ZI1[i][j]

# plot the results

#1
ax1 = plt.subplot(231)
plt.scatter(x1, y1, 10, z1, cmap=cm.jet)
plt.pcolor(XI, YI, ZI1, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

#2
ax2 = plt.subplot(232)
plt.scatter(x2, y2, 10, z2, cmap=cm.jet)
plt.pcolor(XI, YI, ZI2, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

#3
ax3 = plt.subplot(233)
plt.pcolor(XI, YI, ZI5, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

#4
ax4 = plt.subplot(234)
plt.pcolor(XI, YI, ZI6, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

#5
ax5 = plt.subplot(235)
plt.pcolor(XI, YI, ZI7, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

#6
ax6 = plt.subplot(236)
plt.pcolor(XI, YI, ZI8, cmap=cm.jet)
plt.xlim(minX, maxX)
plt.ylim(minY, maxY)
plt.colorbar()

plt.show()
