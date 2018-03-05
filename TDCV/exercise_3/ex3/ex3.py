# -*- coding: UTF-8

import numpy as np
import tensorflow as tf
import quaternion
from matplotlib import pyplot as plt
from functions import *

# generate database set Sdb
#Sdb, Pdb = generate_Sdb()

Strain = generate_Strain()

print(len(Strain))