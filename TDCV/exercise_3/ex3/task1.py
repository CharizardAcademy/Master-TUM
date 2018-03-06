# -*- coding: UTF-8

from functions import *

# generate database set Sdb
Sdb, Pdb = generate_Sdb()

# genetate training set Strain
Strain, Ptrain = generate_Strain()
#print(len(Strain[1][4]))

# generate testing set Stest
Stest,Ptest = generate_Stest()
