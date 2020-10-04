import csv
import os
import numpy as np
import math

def makeDatapathList(rootpath, csv_name):
    csvpath = os.path.join(rootpath, csv_name)
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            for i in range(3, len(row)):
                row[i] = os.path.join(rootpath, row[i])
                camera_angle = 2*math.pi/len(row[3:])*(i-3)
                # camera_angle = -camera_angle    #NED->NEU
                rot_acc_list = rotateVector(row[:3], camera_angle)
                data = rot_acc_list + [row[i]]
                data_list.append(data)
    return data_list

def rotateVector(acc_list, camera_angle):
    acc_numpy = np.array(acc_list)
    acc_numpy = acc_numpy.astype(np.float32)
    rot = np.array([
        [math.cos(-camera_angle), -math.sin(-camera_angle), 0.0],
        [math.sin(-camera_angle), math.cos(-camera_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    rot_acc_numpy = np.dot(rot, acc_numpy)
    rot_acc_list = rot_acc_numpy.tolist()
    return rot_acc_list

##### test #####
# rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/train"
# csv_name = "imu_camera.csv"
# train_list = makeDatapathList(rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3:])
# print("example1: ", train_list[1][:3], train_list[1][3:])
