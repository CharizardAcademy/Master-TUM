# -*- coding: UTF-8

from functions import *
from matplotlib import pyplot as plt
import cv2

# generate database set Sdb
Sdb, Pdb = generate_Sdb()


# genetate training set Strain
Strain, Ptrain = generate_Strain()
#print(len(Strain[1][4]))

# generate testing set Stest
Stest,Ptest = generate_Stest()

# randomly pick an anchor from Strain
anchor,ro = find_anchor(Ptrain)
puller = find_puller(anchor,ro,Pdb)
pusher = find_pusher(anchor,ro,Pdb)










