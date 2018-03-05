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
def read_poses(file):
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


def read_indeces(file):
    index = []
    with open(file) as f:
        for line in f:
            index = line.strip().split(",")
        for i in range(0,len(index)):
            index[i] = int(index[i])
    return index


def read_selected_images(index,path):
    image_list = []
    for i in index:
        image = cv2.imread(path + "coarse" + str(i)+".png")
        image_list.append(image)
    return image_list


def read_selected_poses(index,pose):
    selected_pose = []
    for i in index:
        selected_pose.append(pose[i])
    return selected_pose


def generate_Sdb():
    # generate Sdb set, use all coarse image
    coarse_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/coarse/"

    coarse_ape = read_images(coarse_path + "ape/")
    coarse_benchvise = read_images(coarse_path + "benchvise/")
    coarse_cam = read_images(coarse_path + "cam/")
    coarse_cat = read_images(coarse_path + "cat/")
    coarse_duck = read_images(coarse_path + "/duck")

    coarse_ape_pose = read_poses(coarse_path + "ape/poses.txt")
    coarse_benchvise_pose = read_poses(coarse_path + "ape/poses.txt")
    coarse_cam_pose = read_poses(coarse_path + "ape/poses.txt")
    coarse_cat_pose = read_poses(coarse_path + "ape/poses.txt")
    coarse_duck_pose = read_poses(coarse_path + "ape/poses.txt")

    # Sdb stores the images
    Sdb = [coarse_ape, coarse_benchvise, coarse_cam, coarse_cat, coarse_duck]

    # Pdb stores the poses
    Pdb = [coarse_ape_pose, coarse_benchvise_pose, coarse_cam_pose, coarse_cat_pose, coarse_duck_pose]

    return Sdb, Pdb


def generate_Strain():
    """
    # generate Strain set, use all fine data and selected real data from training_split.txt
    fine_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/fine/"

    fine_ape = read_images(fine_path + "/ape")
    fine_benchvise = read_images(fine_path + "/benchvise")
    fine_cam = read_images(fine_path + "/cam")
    fine_cat = read_images(fine_path + "/cat")
    fine_duck = read_images(fine_path + "/duck")

    fine_ape_pose = read_poses(fine_path + "/ape/poses.txt")
    fine_benchvise_pose = read_poses(fine_path + "/benchvise/poses.txt")
    fine_cam_pose = read_poses(fine_path + "/cam/poses.txt")
    fine_cat_pose = read_poses(fine_path + "/cat/poses.txt")
    fine_duck_pose = read_poses(fine_path + "/duck/poses.txt")

    training_fine = [fine_ape,fine_benchvise,fine_cam,fine_cat,fine_duck]
    training_fine_pose = [fine_ape_pose,fine_benchvise_pose,fine_cam_pose,fine_cat_pose,fine_duck_pose]

    real_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/"
    real_index = read_indeces(real_path + "training_split.txt")

    real_ape = read_selected_images(real_index, real_path + "/ape/")
    real_benchvise = read_selected_images(real_index, real_path + "/benchvise/")
    real_cam = read_selected_images(real_index, real_path + "/cam/")
    real_cat = read_selected_images(real_index, real_path + "/cat/")
    real_duck = read_selected_images(real_index, real_path + "/duck/")
    """

    real_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/"
    real_index = read_indeces(real_path + "training_split.txt")
    real_ape_pose = read_poses(real_path + "/ape/poses.txt")

    final = read_selected_poses(real_index,real_ape_pose)
    #real_benchvise_pose = read_poses(real_path + "/benchvise/poses.txt")
    #real_cam_pose = read_poses(real_path + "/cam/poses.txt")
    #real_cat_pose = read_poses(real_path + "/cat/poses.txt")
    #real_duck_pose = read_poses(real_path + "/duck/poses.txt")

    #training_real = [real_ape,real_benchvise,real_cam,real_cat,real_duck]
    #training_real_pose = [real_ape_pose,real_benchvise_pose,real_cam_pose,real_cat_pose,real_duck_pose]

    #Strain = [training_fine, training_real]
    #Ptrain = [training_fine_pose, training_real_pose]

    return final



