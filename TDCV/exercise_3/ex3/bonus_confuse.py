# coding: UTF-8

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set()
confuse_matrix = np.array([[480,89,42,94,1],[89,428,126,61,2],[89,72,463,82,0],[83,94,167,359,3],[3,55,7,14,627]])
#print(confuse_matrix)

ax = sns.heatmap(confuse_matrix)
plt.show()