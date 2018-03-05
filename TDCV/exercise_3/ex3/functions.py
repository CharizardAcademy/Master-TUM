# -*- coding: UTF-8

import numpy as np
import tensorflow as tf
import quaternion
import cv2
import glob
import os

# read images from folder
def read_images(path):
    image_list = []
    for i in range(0,267):
        image = cv2.imread(path + "coarse" + str(i)+".png")
        image_list.append(image)
    return image_list

# read pose from txt without comments
def read_pose(file):
    q = []
    for line in open(file):
        if line.startswith("#"):
            continue
        else:
            qsub = line.split()
            for i in range(0,len(qsub)):
                qsub[i]=float(qsub[i])
            quat = np.quaternion(qsub[0],qsub[1],qsub[2],qsub[3])
            q.append(quat)
    return q

def generate_Sdb():
    # generate Sdb set, use all coarse image
    coarse_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/coarse/"

    coarse_ape = read_images(coarse_path + "ape/")
    coarse_benchvise = read_images(coarse_path + "benchvise/")
    coarse_cam = read_images(coarse_path + "cam/")
    coarse_cat = read_images(coarse_path + "cat/")
    coarse_duck = read_images(coarse_path + "/duck")

    coarse_ape_pose = read_pose(coarse_path + "ape/poses.txt")
    coarse_benchvise_pose = read_pose(coarse_path + "ape/poses.txt")
    coarse_cam_pose = read_pose(coarse_path + "ape/poses.txt")
    coarse_cat_pose = read_pose(coarse_path + "ape/poses.txt")
    coarse_duck_pose = read_pose(coarse_path + "ape/poses.txt")

    # Sdb stores the images
    Sdb = [coarse_ape, coarse_benchvise, coarse_cam, coarse_cat, coarse_duck]

    # Pdb stores the poses
    Pdb = [coarse_ape_pose, coarse_benchvise_pose, coarse_cam_pose, coarse_cat_pose, coarse_duck_pose]

    return Sdb, Pdb

def generate_Strain():
    #
