import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time

import torch
from torchvision import models
import torch.nn as nn

import sys
sys.path.append('../')

from common_multi import make_datapath_list
from common_combine import data_transform_model
from common import dataset_model
import network
import criterion_model

class Sample:
    def __init__(self,
            index,
            inputs_path, inputs, label, mu, cov, mul_sigma,
            label_r, label_p, output_r, output_p, error_r, error_p):
        self.index = index              #int
        self.inputs_path = inputs_path  #list
        self.inputs = inputs            #ndarray
        self.label = label              #list
        self.mu = mu                    #list
        self.cov = cov                  #ndarray
        self.mul_sigma = mul_sigma      #float
        self.label_r = label_r          #float
        self.label_p = label_p          #float
        self.output_r = output_r        #float
        self.output_p = output_p        #float
        self.error_r = error_r          #float
        self.error_p = error_p          #float

    def printData(self):
        print("-----", self.index, "-----")
        print("inputs_path: ", self.inputs_path)
        # print("inputs: ", self.inputs)
        print("inputs.shape: ", self.inputs.shape)
        print("label: ", self.label)
        print("mu: ", self.mu)
        print("cov: ", self.cov)
        print("mul_sigma: ", self.mul_sigma)
        print("l_r[deg]: ", self.label_r/math.pi*180.0, ", l_p[deg]: ", self.label_p/math.pi*180.0)
        print("o_r[deg]: ", self.output_r/math.pi*180.0, ", o_p[deg]: ", self.output_p/math.pi*180.0)
        print("e_r[deg]: ", self.error_r/math.pi*180.0, ", e_p[deg]: ", self.error_p/math.pi*180.0)

class InferenceModel:
    def __init__(self,
            resize, mean_element, std_element, num_images,
            rootpath, csv_name, batch_size,
            weights_path,
            th_mul_sigma):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.num_images = num_images
        self.data_transform = self.getDataTransform(resize, mean_element, std_element)
        self.datapath_list = []
        self.dataloader = self.getDataloader(rootpath, csv_name, batch_size)
        self.net = self.getNetwork(resize, weights_path)
        ## list
        self.list_samples = []
        self.list_selected_samples = []
        self.list_inputs = []
        self.list_labels = []
        self.list_outputs = []
        self.list_cov = []
        ## threshold
        self.th_mul_sigma = th_mul_sigma

    def getDataTransform(self, resize, mean_element, std_element):
        mean = ([mean_element, mean_element, mean_element])
        std = ([std_element, std_element, std_element])
        if self.num_images > 0:
            data_transform = data_transform_model.DataTransform(resize, mean, std, self.num_images)
        else:
            data_transform = data_transform_model.DataTransform(resize, mean, std)
        return data_transform

    def getDataloader(self, rootpath, csv_name, batch_size):
        ## list
        self.datapath_list = make_datapath_list.makeDatapathList(rootpath, csv_name)
        ## get number of input images
        if self.num_images < 0:
            self.num_images = len(self.datapath_list[0][3:])
        print("self.num_images = ", self.num_images)
        ## dataset
        dataset = dataset_model.OriginalDataset(
            data_list=self.datapath_list,
            transform=self.data_transform,
            phase="val"
        )
        ## dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return dataloader

    def getNetwork(self, resize, weights_path):
        net = network.OriginalNet(self.num_images, resize=resize, use_pretrained=False)
        print(net)
        net.to(self.device)
        net.eval()
        ## load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("Loaded [GPU -> GPU]: ", weights_path)
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> CPU]: ", weights_path)
        net.load_state_dict(loaded_weights)
        return net

    def infer(self):
        ## time
        start_clock = time.time()
        ## data load
        loss_all = 0.0
        for inputs, labels in tqdm(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ## compute gradient
            with torch.set_grad_enabled(False):
                ## forward
                outputs = self.net(inputs)
                loss_batch, cov = self.computeLossAndCov(outputs, labels)
                ## add loss
                loss_all += loss_batch.item() * inputs.size(0)
                # print("loss_batch.item() = ", loss_batch.item())
            ## append
            self.list_inputs += list(inputs.cpu().detach().numpy())
            self.list_labels += labels.cpu().detach().numpy().tolist()
            self.list_outputs += outputs.cpu().detach().numpy().tolist()
            self.list_cov += list(cov.cpu().detach().numpy())
        ## average loss
        loss_all = loss_all / len(self.dataloader.dataset)
        print("Loss: {:.4f}".format(loss_all))
        ## compute error
        mae, var, ave_mul_std, selected_mae, selected_var, weighted_mae = self.computeAttitudeError()
        ## sort
        self.sortSamples()
        ## show result & set graph
        self.showResult()
        print ("-----")
        ## inference time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("inference time: ", mins, " [min] ", secs, " [sec]")
        ## MAE & Var
        print("mae [deg] = ", mae)
        print("var [deg^2] = ", var)
        ## average multiplied sigma
        print("ave_mul_std [m^3/s^6] = ", ave_mul_std)
        ## selected MAE & Var
        print("th_mul_sigma = ", self.th_mul_sigma)
        print("number of the selected samples = ", len(self.list_selected_samples), " / ", len(self.list_samples))
        print("selected mae [deg] = ", selected_mae)
        print("selected var [deg^2] = ", selected_var)
        print("weighted mae [deg] = ", weighted_mae)
        ## graph
        plt.tight_layout()
        plt.show()

    def computeLossAndCov(self, outputs, labels):
        criterion = criterion_model.CriterionModel()
        loss = criterion.computeLoss(outputs, labels, self.device)
        cov = criterion.getCovMatrix(outputs)
        return loss, cov

    def computeAttitudeError(self):
        list_errors = []
        list_selected_errors = []
        list_mul_std = []
        for i in range(len(self.list_labels)):
            ## error
            label_r, label_p = self.accToRP(self.list_labels[i])
            output_r, output_p = self.accToRP(self.list_outputs[i])
            error_r = self.computeAngleDiff(output_r, label_r)
            error_p = self.computeAngleDiff(output_p, label_p)
            list_errors.append([error_r, error_p])
            ## multiplied sigma
            mul_sigma = math.sqrt(self.list_cov[i][0, 0]) * math.sqrt(self.list_cov[i][1, 1]) * math.sqrt(self.list_cov[i][2, 2])
            list_mul_std.append(mul_sigma)
            ## register
            sample = Sample(
                i,
                self.datapath_list[i][3:], self.list_inputs[i], self.list_labels[i], self.list_outputs[i], self.list_cov[i], mul_sigma,
                label_r, label_p, output_r, output_p, error_r, error_p
            )
            self.list_samples.append(sample)
            ## judge
            if mul_sigma < self.th_mul_sigma:
                self.list_selected_samples.append(sample)
                list_selected_errors.append([error_r, error_p])
        arr_errors = np.array(list_errors)
        arr_selected_errors = np.array(list_selected_errors)
        print("arr_errors.shape = ", arr_errors.shape)
        mae = self.computeMAE(arr_errors/math.pi*180.0)
        var = self.computeVar(arr_errors/math.pi*180.0)
        ave_mul_std = np.mean(list_mul_std, axis=0)
        selected_mae = self.computeMAE(arr_selected_errors/math.pi*180.0)
        selected_var = self.computeVar(arr_selected_errors/math.pi*180.0)
        list_weighted_error = list(np.array(list_errors)/math.pi*180.0 * (1/np.array(list_mul_std)[:, np.newaxis]))
        weighted_mae = np.sum(np.abs(list_weighted_error), axis=0) / np.sum(1/np.array(list_mul_std))
        return mae, var, ave_mul_std, selected_mae, selected_var, weighted_mae

    def accToRP(self, acc):
        r = math.atan2(acc[1], acc[2])
        p = math.atan2(-acc[0], math.sqrt(acc[1]*acc[1] + acc[2]*acc[2]))
        return r, p

    def computeAngleDiff(self, angle1, angle2):
        diff = math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2))
        return diff

    def computeMAE(self, x):
        return np.mean(np.abs(x), axis=0)

    def computeVar(self, x):
        return np.var(x, axis=0)

    def sortSamples(self):
        list_sum_error_rp = [abs(sample.error_r) + abs(sample.error_p) for sample in self.list_samples]
        list_mul_std = [sample.mul_sigma for sample in self.list_samples]
        ## get indicies
        sorted_indicies = np.argsort(list_sum_error_rp)         #error: small->large
        # sorted_indicies = np.argsort(list_sum_error_rp)[::-1]   #error: large->small
        # sorted_indicies = np.argsort(list_mul_std)            #sigma: small->large
        # sorted_indicies = np.argsort(list_mul_std)[::-1]      #sigma: large->small
        ## sort
        self.list_samples = [self.list_samples[index] for index in sorted_indicies]

    def showResult(self):
        plt.figure()
        h = 5
        w = 10
        for i in range(len(self.list_samples)):
            self.list_samples[i].printData()
            if i < h*w:
                plt.subplot(h, w, i+1)
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(np.clip(self.list_samples[i].inputs.transpose((1, 2, 0)), 0, 1))
                plt.title(str(self.list_samples[i].index))

def main():
    ## hyperparameters
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    num_images = -1
    rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/val"
    csv_name = "imu_camera.csv"
    batch_size = 10
    weights_path = "../../weights/mle.pth"
    th_mul_sigma = 0.0001
    ## infer
    inference_model = InferenceModel(
        resize, mean_element, std_element, num_images,
        rootpath, csv_name, batch_size,
        weights_path,
        th_mul_sigma
    )
    inference_model.infer()

if __name__ == '__main__':
    main()
