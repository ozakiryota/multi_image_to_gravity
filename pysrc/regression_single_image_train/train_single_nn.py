import sys
sys.path.append('../regression')

import train    #from ../regression

def main():
    ## hyperparameters
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    num_images = -1
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/val"
    csv_name = "imu_camera.csv"
    batch_size = 10
    str_optimizer = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-5
    lr_fc = 1e-4
    num_epochs = 50
    ## train
    train_model = train.TrainModel(
        resize, mean_element, std_element, num_images,
        train_rootpath, val_rootpath, csv_name, batch_size,
        str_optimizer, lr_cnn, lr_fc,
        num_epochs
    )
    train_model.train()

if __name__ == '__main__':
    main()
