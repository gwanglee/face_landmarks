#-*- coding: utf-8 -*-

'''
AR Core app 에서 얻은 landmark point를 display 해보자
'''

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

FILEPATH = '/Users/gglee/Downloads/mesh_09.txt'

with open(FILEPATH) as f:
    for l in f.readlines():
        words = l.split()
        nums = [float(w.strip()) for w in words]

        landmarks = np.reshape(np.asarray(nums, dtype=np.float32), (-1, 3))
        # print(landmarks, np.amax(landmarks), np.amin(landmarks))

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter(landmarks[:, 2], landmarks[:, 0], landmarks[:, 1])
        fig.show()
        plt.show()