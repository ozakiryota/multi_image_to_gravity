from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class OriginalNet(nn.Module):
    def __init__(self, num_images, resize=224, use_pretrained=True):
        super(OriginalNet, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained)
        self.features = vgg.features
        if num_images == 5 and resize == 112:
            num_fc_in_features = 26112
        elif num_images == 5:
            num_fc_in_features = 125440
        elif num_images == 4 and resize == 112:
            num_fc_in_features = 21504
        elif num_images == 4:
            num_fc_in_features = 100352
        elif num_images == 1 and resize == 112:
            num_fc_in_features = 4608
        elif num_images == 1:
            num_fc_in_features = 25088
        self.fc = nn.Sequential(
            nn.Linear(num_fc_in_features, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, 3)
        )

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "features" in param_name:
                # print("features: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value

    def forward(self, x):
        # print("cnn-in", x.size())
        x = self.features(x)
        # print("cnn-out", x.size())
        x = torch.flatten(x, 1)
        # print("fc-in", x.size())
        x = self.fc(x)
        # print("fc-out", x.size())
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x

##### test #####
# import sys
# sys.path.append('../')
# from common_combine import data_transform_model
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
# # print(list_fc_param_value)
# ## trans param
# resize = 224
# # resize = 112
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = data_transform_model.DataTransform(resize, mean, std)
# img_trans, _ = transform(img_path_list, acc_numpy)
# ## network
# net = OriginalNet(len(img_path_list), resize)
# print(net)
# list_cnn_param_value, list_fc_param_value = net.getParamValueList()
# ## prediction
# inputs = img_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
