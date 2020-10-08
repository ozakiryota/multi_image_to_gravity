from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')

from common_single import make_datapath_list
from common_single import data_transform_model
from common import dataset_model
from combine_regression import network

class TrainModel:
    def __init__(self,
            resize, mean_element, std_element, num_images,
            train_rootpath, val_rootpath, csv_name, batch_size,
            str_optimizer, lr_cnn, lr_fc,
            num_epochs,
            weights_path):
        self.setRandomCondition()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.num_images = num_images
        self.data_transform = self.getDataTransform(resize, mean_element, std_element)
        self.dataloaders_dict = self.getDataloader(train_rootpath, val_rootpath, csv_name, batch_size)
        self.net = self.getNetwork(resize, weights_path)
        self.optimizer = self.getOptimizer(str_optimizer, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter  = self.getStrHyperparameter(resize, mean_element, std_element, str_optimizer, lr_cnn, lr_fc, batch_size)

    def setRandomCondition(self, keep_reproducibility=False):
        if keep_reproducibility:
            torch.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataTransform(self, resize, mean_element, std_element):
        mean = ([mean_element, mean_element, mean_element])
        std = ([std_element, std_element, std_element])
        if self.num_images > 0:
            data_transform = data_transform_model.DataTransform(resize, mean, std, self.num_images)
        else:
            data_transform = data_transform_model.DataTransform(resize, mean, std)
        return data_transform

    def getDataloader(self, train_rootpath, val_rootpath, csv_name, batch_size):
        ## list
        train_list = make_datapath_list.makeDatapathList(train_rootpath, csv_name)
        val_list = make_datapath_list.makeDatapathList(val_rootpath, csv_name)
        ## get number of input images
        if self.num_images < 0:
            self.num_images = len(train_list[0][3:])
        print("self.num_images = ", self.num_images)
        ## dataset
        train_dataset = dataset_model.OriginalDataset(
            data_list=train_list,
            transform=self.data_transform
        )
        val_dataset = dataset_model.OriginalDataset(
            data_list=val_list,
            transform=self.data_transform
        )
        ## dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        return dataloaders_dict

    def getNetwork(self, resize, weights_path):
        net = network.OriginalNet(self.num_images, resize=resize, use_pretrained=False)
        print(net)
        net.to(self.device)
        ## load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("Loaded [GPU -> GPU]: ", weights_path)
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> CPU]: ", weights_path)
        net.load_state_dict(loaded_weights)
        return net

    def getOptimizer(self, str_optimizer, lr_cnn, lr_fc):
        ## param
        list_cnn_param_value, list_fc_param_value = self.net.getParamValueList()
        ## optimizer
        if str_optimizer == "SGD":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc}
            ], momentum=0.9)
        elif str_optimizer == "Adam":
            optimizer = optim.Adam([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc}
            ])
        print(optimizer)
        return optimizer

    def getStrHyperparameter(self, resize, mean_element, std_element, str_optimizer, lr_cnn, lr_fc, batch_size):
        str_hyperparameter = "regression" \
            + str(self.num_images) + "images" \
            + str(len(self.dataloaders_dict["train"].dataset)) + "finetune" \
            + str(len(self.dataloaders_dict["val"].dataset)) + "val" \
            + str(resize) + "resize" \
            + str(mean_element) + "mean" \
            + str(std_element) + "std" \
            + str_optimizer \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter

    def train(self):
        ## time
        start_clock = time.time()
        ## loss record
        # writer = SummaryWriter(logdir="../../runs")
        record_loss_train = []
        record_loss_val = []
        ## loop
        for epoch in range(self.num_epochs):
            print("----------")
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))
            ## phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.net.train()
                else:
                    self.net.eval()
                ## skip
                if (epoch == 0) and (phase=="train"):
                    continue
                ## data load
                epoch_loss = 0.0
                for inputs, labels in tqdm(self.dataloaders_dict[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    ## reset gradient
                    self.optimizer.zero_grad()   #reset grad to zero (after .step())
                    ## compute gradient
                    with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                        ## forward
                        outputs = self.net(inputs)
                        loss = self.computeLoss(outputs, labels)
                        ## backward
                        if phase == "train":
                            loss.backward()     #accumulate gradient to each Tensor
                            self.optimizer.step()    #update param depending on current .grad
                        ## add loss
                        epoch_loss += loss.item() * inputs.size(0)
                        # print("loss.item() = ", loss.item())
                ## average loss
                epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
                ## record
                if phase == "train":
                    record_loss_train.append(epoch_loss)
                    # writer.add_scalar("Loss/train", epoch_loss, epoch)
                else:
                    record_loss_val.append(epoch_loss)
                    # writer.add_scalar("Loss/val", epoch_loss, epoch)
                # writer.close()
        ## save
        self.saveParam()
        self.saveGraph(record_loss_train, record_loss_val)
        ## training time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("training_time: ", mins, " [min] ", secs, " [sec]")

    def computeLoss(self, outputs, labels):
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        return loss

    def saveParam(self):
        save_path = "../../weights/" + self.str_hyperparameter + ".pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved: ", save_path)

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m^2/s^4]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig("../../graph/" + self.str_hyperparameter + ".jpg")
        plt.show()

def main():
    ## hyperparameters
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    num_images = -1
    train_rootpath = "../../../dataset_image_to_gravity/stick/campus_cww"
    val_rootpath = "../../../dataset_image_to_gravity/stick/dkan_outdoor"
    csv_name = "imu_camera.csv"
    batch_size = 10
    str_optimizer = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-6
    lr_fc = 1e-5
    num_epochs = 50
    weights_path = "../../weights/regression1cam.pth"
    ## train
    train_model = TrainModel(
        resize, mean_element, std_element, num_images,
        train_rootpath, val_rootpath, csv_name, batch_size,
        str_optimizer, lr_cnn, lr_fc,
        num_epochs,
        weights_path
    )
    train_model.train()

if __name__ == '__main__':
    main()
