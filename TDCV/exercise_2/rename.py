# -*- coding:UTF-8

import os

path = "/users/gaoyingqiang/Desktop/data/task3/train/04/rotate/"
counter = 0
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file))==True:
        newname = str(counter)+'.jpg'
        counter = counter+1
        os.rename(os.path.join(path,file),os.path.join(path,newname))
        print("OK")
