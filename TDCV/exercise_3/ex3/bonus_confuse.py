# coding: UTF-8

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set()
confuse_matrix_9 = np.array([[480,89,42,94,1],[89,428,126,61,2],[89,72,463,82,0],[83,94,167,359,3],[3,55,7,14,627]])
#print(confuse_matrix)
confuse_matrix_8 = np.array([[443,106,112,45,0],[58,428,183,30,7],[36,82,505,83,0],[28,84,183,408,3],[2,35,9,3,657]])
ax = sns.heatmap(confuse_matrix_8,annot=True,fmt="d")
plt.show()