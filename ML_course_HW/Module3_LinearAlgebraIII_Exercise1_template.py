from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def draw_plane(A, b):
    xx, yy = np.meshgrid(np.arange(-10,10.1,0.1), np.arange(-10,10.1,0.1))
    plt3d = plt.figure().gca(projection='3d')
    for i in range(3):
      a, c, d, e = A[i][0], A[i][1], A[i][2], b[i]
      zz = (-a*xx - c*yy + e) / d
      plt3d.plot_surface(xx, yy, zz, alpha=0.5, cmap=cm.coolwarm)
    ax = plt.gca()
    #z = np.linalg.solve(A, b)
    #ax.scatter(z[0] , z[1] , z[2],  color='red')
    plt.show()

def show_properties(A, b):
    print('  rank(A):', np.linalg.matrix_rank(A))
    print('  rank([A, b]):', np.linalg.matrix_rank(np.hstack((A,b))))
    if np.linalg.matrix_rank(A) == 3:
      z = np.linalg.solve(A,b)
      print('  residual: {:.2f}'.format(np.linalg.norm(A.dot(z)-b))) # ||A*z-b||
      print('  solution:', z)
    print('  cond(A): {:.2e}'.format(np.linalg.cond(A)))

if __name__ == '__main__':
  print('Exercise 1.1')
  A = np.array([[3, 2, -1], [6, -1, 3], [1, 10, -2]])
  b = np.array([[-7], [-4], [2]])
  #draw_plane(A, b)
  show_properties(A, b)
  print()

  print('Exercise 1.2')
  A = np.array([[4, -1, 3], [21, -4, 18], [-9, 1, -9]])
  b = np.array([[5], [7], [-8]])
  #draw_plane(A, b)
  show_properties(A, b)
  print()

  print('Exercise 1.3')
  A = np.array([[7, -4, 1], [3, 2, -1], [5, 12, -5]])
  b = np.array([[-15], [-5], [-5]])
  #draw_plane(A, b)
  show_properties(A, b)
