import sys
sys.path.append('../regression')

import inference    #from ../regression

def main():
    ## hyperparameters
    resize = 224
    mean_element = 0.5
    std_element = 0.5
    num_images = -1
    rootpath = "../../../dataset_image_to_gravity/AirSim/5cam/val"
    csv_name = "imu_camera.csv"
    batch_size = 10
    weights_path = "../../weights/regression1cam.pth"
    ## infer
    inference_model = inference.InferenceModel(
        resize, mean_element, std_element, num_images,
        rootpath, csv_name, batch_size,
        weights_path
    )
    inference_model.infer()

if __name__ == '__main__':
    main()
