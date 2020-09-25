from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std):
        self.DataTransform = transforms.Compose([
            transforms.Resize(resize),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_path_list, acc_numpy):
        combined_img_pil = self.combineImages(img_path_list)
        img_tensor = self.DataTransform(combined_img_pil)
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)

        return img_tensor, acc_tensor

    def combineImages(self, img_path_list):
        for i in range(len(img_path_list)):
            if i == 0:
                combined_img_pil = Image.open(img_path_list[i])
            else:
                combined_img_pil = self.getConcatH(combined_img_pil, Image.open(img_path_list[i]))
        return combined_img_pil

    def getConcatH(self, img_l, img_r):
        dst = Image.new('RGB', (img_l.width + img_r.width, img_l.height))
        dst.paste(img_l, (0, 0))
        dst.paste(img_r, (img_l.width, 0))
        return dst

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
# transform = DataTransform(resize, mean, std)
# img_trans, acc_trans = transform(img_path_list, acc_numpy)
# print("acc_trans", acc_trans)
# ## tensor -> numpy
# img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
# print("img_trans_numpy.shape = ", img_trans_numpy.shape)
# ## save
# save_path = "../../keep/augmented_example.jpg"
# img_pil = Image.fromarray(np.uint8(255*img_trans_numpy))
# img_pil.save(save_path)
# print("saved: ", save_path)
# ## imshow
# for i in range(len(img_path_list)):
#     plt.subplot2grid((2, len(img_path_list)), (0, i))
#     plt.imshow(Image.open(img_path_list[i]))
# plt.subplot2grid((2, len(img_path_list)), (1, 0), colspan=len(img_path_list))
# plt.imshow(img_trans_numpy)
# plt.show()
