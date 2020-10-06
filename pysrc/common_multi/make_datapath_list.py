import csv
import os

def makeDatapathList(rootpath, csv_name):
    csvpath = os.path.join(rootpath, csv_name)
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            for i in range(3, len(row)):
                row[i] = os.path.join(rootpath, row[i])
            data_list.append(row)
    return data_list

##### test #####
# rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/train"
# csv_name = "imu_camera.csv"
# train_list = makeDatapathList(rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3:])
# print("example1: ", train_list[1][:3], train_list[1][3:])
