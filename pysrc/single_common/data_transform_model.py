from PIL import Image, ImageOps
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

    def __call__(self, img_path_list, acc_numpy):
        angle_deg = random.uniform(-10.0, 10.0)
        angle_rad = angle_deg / 180 * math.pi
        ## image
        img_pil = Image.open(img_path_list[0])
        is_mirror = bool(random.getrandbits(1))
        print(is_mirror)
        if is_mirror:
            img_pil = ImageOps.mirror(img_pil)
        rot_img_pil = img_pil.rotate(angle_deg)
        rot_img_tensor = self.img_transform(rot_img_pil)
        ## acc
        if is_mirror:
            acc_numpy[1] = -acc_numpy[1]
        rot_acc_numpy = self.rotateVector(acc_numpy, angle_rad)
        rot_acc_numpy = rot_acc_numpy.astype(np.float32)
        rot_acc_numpy = rot_acc_numpy / np.linalg.norm(rot_acc_numpy)
        rot_acc_tensor = torch.from_numpy(rot_acc_numpy)
        return rot_img_tensor, rot_acc_tensor

    def rotateVector(self, acc_numpy, angle):
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(-angle), -math.sin(-angle)],
            [0, math.sin(-angle), math.cos(-angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

##### test #####
## trans param
resize = 224
mean = ([0.5, 0.5, 0.5])
std = ([0.5, 0.5, 0.5])
## image
img_path_list = [
    "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_0.jpg"
    # "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_288.jpg",
    # "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_216.jpg",
    # "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_144.jpg",
    # "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_72.jpg"
]
## label
acc_list = [0, 0, 1]
acc_numpy = np.array(acc_list)
## transform
transform = DataTransform(resize, mean, std)
img_trans, acc_trans = transform(img_path_list, acc_numpy)
print("acc_trans = ", acc_trans)
## tensor -> numpy
img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
print("img_trans_numpy.shape = ", img_trans_numpy.shape)
## save
save_path = "../../keep/augmented_single_example.jpg"
img_pil = Image.fromarray(np.uint8(255*img_trans_numpy))
img_pil.save(save_path)
print("saved: ", save_path)
## imshow
for i in range(len(img_path_list)):
    plt.subplot2grid((2, len(img_path_list)), (0, len(img_path_list)-i-1))
    plt.imshow(Image.open(img_path_list[i]))
plt.subplot2grid((2, len(img_path_list)), (1, 0), colspan=len(img_path_list))
plt.imshow(img_trans_numpy)
plt.show()
