from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import linalg

def draw_surface(A, b):
  xx, yy = np.meshgrid(np.arange(-2,2.1,0.1), np.arange(-2,2.1,0.1))
  F1 = A[0][0]*xx + A[0][1]*yy - b[0];
  F2 = A[1][0]*xx + A[1][1]*yy - b[1];
  zz = F1*F1 + F2*F2;
  plt3d = plt.figure().gca(projection='3d')
  plt3d.plot_surface(xx, yy, zz, alpha=0.7, cmap=cm.coolwarm)

if __name__ == '__main__':
  b1 = 1; b2 = 2
  b = np.array([[b1], [b2]])
  a11 = 1; a12 = -2; a21 = 1; a22 = -2
  A = np.array([[a11, a12], [a21, a22]])
  AtA = np.dot(A.T, A)
  Atb = np.dot(A.T, b)

  x = np.array([[-2], [2]]) # initial position
  draw_surface(A, b)

  ax = plt.gca()
  z = linalg.norm(np.dot(A, x) - b) ** 2 # || Ax-b ||^2
  ax.scatter(x[0] , x[1] , z,  color='red')

  # insert your code here

  plt.show()
