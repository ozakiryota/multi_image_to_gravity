from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std):
        self.img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_path_list, acc_numpy, phase):
        img_tensor_list = self.transformImages(img_path_list)
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return img_tensor_list, acc_tensor

    def transformImages(self, img_path_list):
        img_tensor_list = []
        for i in range(len(img_path_list)):
            img_tensor = self.img_transform(Image.open(img_path_list[i]))
            img_tensor_list.append(img_tensor)
        return img_tensor_list

##### test #####
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## image
# img_path_list = [
#     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_0.jpg",
#     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_72.jpg",
#     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_144.jpg",
#     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_216.jpg",
#     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_288.jpg"
# ]
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## transform
# data_transform = DataTransform(resize, mean, std)
# img_trans_list, acc_trans = data_transform(img_path_list, acc_numpy)
# print("acc_trans = ", acc_trans)
# ## tensor -> numpy
# img_trans_numpy_list = [np.clip(img_trans.numpy().transpose((1, 2, 0)), 0, 1) for img_trans in img_trans_list]  #(rgb, h, w) -> (h, w, rgb)
# print("np.array(img_trans_numpy_list).shape = ", np.array(img_trans_numpy_list).shape)
# ## imshow
# for i in range(len(img_path_list)):
#     plt.subplot2grid((2, len(img_path_list)), (0, i))
#     plt.imshow(Image.open(img_path_list[i]))
#     plt.subplot2grid((2, len(img_path_list)), (1, i))
#     plt.imshow(img_trans_numpy_list[i])
# plt.show()
