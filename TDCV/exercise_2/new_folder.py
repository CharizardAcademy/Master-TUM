# -*- coding:UTF-8

import os

for x in range(0,44):
    path = "/users/gaoyingqiang/Desktop/data/task3/test/"+str(x)+"/"
    if os.path.isdir(path + str(50)):
        pass
    else:
        os.mkdir(path + str(50))
    if os.path.isdir(path + str(100)):
        pass
    else:
        os.mkdir(path + str(100))
    if os.path.isdir(path + str(150)):
        pass
    else:
        os.mkdir(path + str(150))
