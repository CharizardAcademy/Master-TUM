# -*- coding: UTF-8

from functions import *
from matplotlib import pyplot as plt
import cv2
import scipy as sp

# generate database set Sdb
Sdb, Pdb = generate_Sdb()
#print(len(Sdb[0]))

# genetate training set Strain
Strain, Ptrain = generate_Strain()
#print(len(Strain[1][0]))

# generate testing set Stest
Stest,Ptest = generate_Stest()

batch = batch_generator(10,Ptrain,Pdb,Strain,Sdb)

"""
cv2.imshow("anchor",batch[0])
cv2.imshow("puller",batch[1])
cv2.imshow("pusher",batch[2])
cv2.waitKey(0)
"""













