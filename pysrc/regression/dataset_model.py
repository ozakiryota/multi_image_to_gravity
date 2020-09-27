import torch.utils.data as data
from PIL import Image
import numpy as np

import torch

class OriginalDataset(data.Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path_list = self.data_list[index][3:]
        acc_str_list = self.data_list[index][:3]
        acc_list = [float(num) for num in acc_str_list]
        acc_numpy = np.array(acc_list)

        img_trans, acc_trans = self.transform(img_path_list, acc_numpy)

        return img_trans, acc_trans

##### test #####
# import make_datapath_list
# import data_transform_model
# ## list
# train_rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/train"
# val_rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/val"
# csv_name = "imu_camera.csv"
# train_list = make_datapath_list.makeDatapathList(train_rootpath, csv_name)
# val_list = make_datapath_list.makeDatapathList(val_rootpath, csv_name)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## dataset
# train_dataset = OriginalDataset(
#     data_list=train_list,
#     transform=data_transform_model.DataTransform(resize, mean, std)
# )
# val_dataset = OriginalDataset(
#     data_list=val_list,
#     transform=data_transform_model.DataTransform(resize, mean, std)
# )
# ## print
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].size())   #data
# print("index", index, ": ", train_dataset.__getitem__(index)[1])   #label
