# -*- coding: UTF-8

import numpy as np
import cv2
import quaternion
import random


# read images from folder
def read_images(path,datatype):
    if datatype == "coarse":
        length_index = 0
    elif datatype == "fine":
        length_index = 1
    else:
        length_index = 2
    image_list = []
    image_length = [267,1011,1178]
    for i in range(0,image_length[length_index]):
        image = cv2.imread(path + datatype + str(i)+".png")
        image = image - np.average(image)
        image = image/np.std(image)
        image_list.append(image)

    return image_list


# read pose from txt without comments
def read_poses(file):
    q = []
    for line in open(file):
        if line.startswith("#"):
            continue
        elif line.startswith("\n"):
            break
        else:
            qsub = line.split()
            for i in range(0, len(qsub)):
                qsub[i] = float(qsub[i])
            q.append(qsub)
    #q = np.array(q)

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
        image = cv2.imread(path + "real" + str(i)+".png")
        #image = image - np.mean(image)
        #image = image / np.std(image, 0, 1)
        image_list.append(image)

    return image_list


def read_selected_poses(index,path):
    selected_pose = []
    pose = read_poses(path)
    for i in index:
        selected_pose.append(pose[i])

    return selected_pose


def generate_Sdb():
    # generate Sdb set, use all coarse image
    coarse_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/coarse/"

    coarse_ape = read_images(coarse_path + "ape/","coarse")
    coarse_benchvise = read_images(coarse_path + "benchvise/","coarse")
    coarse_cam = read_images(coarse_path + "cam/","coarse")
    coarse_cat = read_images(coarse_path + "cat/","coarse")
    coarse_duck = read_images(coarse_path + "duck/","coarse")

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
    # generate Strain set, use all fine data and selected real data from training_split.txt
    fine_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/fine/"

    fine_ape = read_images(fine_path + "/ape/","fine")
    fine_benchvise = read_images(fine_path + "/benchvise/","fine")
    fine_cam = read_images(fine_path + "/cam/","fine")
    fine_cat = read_images(fine_path + "/cat/","fine")
    fine_duck = read_images(fine_path + "/duck/","fine")

    fine_ape_pose = read_poses(fine_path + "/ape/poses.txt")
    fine_benchvise_pose = read_poses(fine_path + "/benchvise/poses.txt")
    fine_cam_pose = read_poses(fine_path + "/cam/poses.txt")
    fine_cat_pose = read_poses(fine_path + "/cat/poses.txt")
    fine_duck_pose = read_poses(fine_path + "/duck/poses.txt")

    training_fine = [fine_ape,fine_benchvise,fine_cam,fine_cat,fine_duck]
    training_fine_pose = [fine_ape_pose,fine_benchvise_pose,fine_cam_pose,fine_cat_pose,fine_duck_pose]

    real_path = "/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/"
    real_index = read_indeces(real_path + "training_split.txt")

    real_ape = read_selected_images(real_index, real_path + "ape/")
    real_benchvise = read_selected_images(real_index, real_path + "benchvise/")
    real_cam = read_selected_images(real_index, real_path + "cam/")
    real_cat = read_selected_images(real_index, real_path + "cat/")
    real_duck = read_selected_images(real_index, real_path + "duck/")

    real_ape_pose = read_selected_poses(real_index,real_path + "ape/poses.txt")
    real_benchvise_pose = read_selected_poses(real_index,real_path + "benchvise/poses.txt")
    real_cam_pose = read_selected_poses(real_index,real_path + "cam/poses.txt")
    real_cat_pose = read_selected_poses(real_index,real_path + "cat/poses.txt")
    real_duck_pose = read_selected_poses(real_index,real_path + "duck/poses.txt")

    training_real = [real_ape,real_benchvise,real_cam,real_cat,real_duck]
    training_real_pose = [real_ape_pose,real_benchvise_pose,real_cam_pose,real_cat_pose,real_duck_pose]

    Strain = [training_fine, training_real]
    Ptrain = [training_fine_pose, training_real_pose]

    return Strain, Ptrain


def test_index():
    index = range(1177)
    train_index = read_indeces("/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/training_split.txt")
    test_index = list(set(index).difference(set(train_index)))
    return test_index


def generate_Stest():
    index = test_index()
    test_ape = read_selected_images(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/ape/")
    test_benchvise = read_selected_images(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/benchvise/")
    test_cam = read_selected_images(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/cam/")
    test_cat = read_selected_images(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/cat/")
    test_duck = read_selected_images(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/duck/")

    test_ape_pose= read_selected_poses(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/ape/poses.txt")
    test_benchvise_pose = read_selected_poses(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/benchvise/poses.txt")
    test_cam_pose = read_selected_poses(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/cam/poses.txt")
    test_cat_pose = read_selected_poses(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/cat/poses.txt")
    test_duck_pose = read_selected_poses(index,"/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/dataset/real/duck/poses.txt")

    Stest = [test_ape,test_benchvise,test_cam,test_cat,test_duck]
    Ptest = [test_ape_pose,test_benchvise_pose,test_cam_pose,test_cat_pose,test_duck_pose]

    return Stest, Ptest


def random_generator():
    # generate random index for fine and real data
    random_dataclass = random.sample([0,1],1)
    # generate random index for object
    random_objectclass = random.sample([0,1,2,3,4],1)
    # generate random index for image
    if random_dataclass == 0: # fine data
        random_index = random.sample(range(0,1011),1)
    else: # real data
        random_index = random.sample(range(0,471),1)

    return random_dataclass,random_objectclass,random_index


def find_anchor(Ptrain,Strain):
    random_data, random_obj, random_index = random_generator()
    rd = random_data[0]
    ro = random_obj[0]
    ri = random_index[0]
    anchor_pose = Ptrain[rd][ro][ri]
    anchor_image = Strain[rd][ro][ri]

    return anchor_pose,anchor_image,ro


def find_puller(anchor_pose,ro,Pdb,Sdb):
    puller_bank = []
    for i in range(0, len(Pdb[ro])):
        quat = Pdb[ro][i]
        qmulti = np.abs(np.inner(anchor_pose, quat))
        theta = 2 * np.arccos(qmulti)
        puller_bank.append(theta)

    min_theta = np.min(puller_bank)
    puller_pose = Pdb[ro][puller_bank.index(min_theta)]
    puller_image = Sdb[ro][puller_bank.index(min_theta)]

    return puller_pose,puller_image

# the pusher is defined as combination of both assumptions
def find_pusher(anchor_pose,ro,Pdb,Sdb):
    pusher_bank = []
    # randomly pick another object different to anchor image
    pusher_random_obj = random.sample(list(set(range(0,5)).difference(set([ro]))),1)[0]
    for i in range(0, len(Pdb[pusher_random_obj])):
        quat = Pdb[pusher_random_obj][i]
        qmulti = np.abs(np.inner(anchor_pose, quat))
        theta = 2 * np.arccos(qmulti)
        pusher_bank.append(theta)

    max_theta = np.max(pusher_bank)
    pusher_pose = Pdb[pusher_random_obj][pusher_bank.index(max_theta)]
    pusher_image = Sdb[pusher_random_obj][pusher_bank.index(max_theta)]

    return pusher_pose,pusher_image


def batch_generator(n,Ptrain,Pdb,Strain,Sdb):
    batch = []
    for i in range(0,n):
        anchor_pose,anchor_image,ro = find_anchor(Ptrain,Strain)
        _,puller_image = find_puller(anchor_pose,ro,Pdb,Sdb)
        _,pusher_image = find_pusher(anchor_pose,ro,Pdb,Sdb)
        batch.append([anchor_image,puller_image,pusher_image])

    return batch
















