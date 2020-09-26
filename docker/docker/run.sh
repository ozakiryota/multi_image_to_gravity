#!/bin/bash

image_name="multi_image_to_gravity"
tag_name="docker"
root_path=$(pwd)

xhost +
docker run -it --rm \
	--gpus all \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-v $root_path/../../../dataset_image_to_gravity:/home/$image_name/../dataset_image_to_gravity \
	-v $root_path/../../weights:/home/$image_name/weights \
	-v $root_path/../../graph:/home/$image_name/graph \
	$image_name:$tag_name
