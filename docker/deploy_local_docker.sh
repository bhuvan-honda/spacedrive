#!/bin/bash

# DATADIR=<path_to_your_data_directory>
DATADIR="data/nuscenes"
# WORKDIR=<path_to_your_code_directory>
WORKDIR="/home/J0986625/SpaceDrive/"
# CKPTDIR=<path_to_your_checkpoint_directory>
CKPTDIR="ckpts"

PROJECT_NAME="spacedrive"

docker run \
    --user $USER \
    --gpus all \
    --mount type=bind,source=$WORKDIR,target=/workspace \
    --mount type=bind,source=/mnt,target=/mnt \
    --shm-size 256g \
    -e PROJECT_NAME=$PROJECT_NAME \
    -e HTTP_PROXY="http://172.17.0.1:3128" \
    -e HTTPS_PROXY="http://172.17.0.1:3128" \
    -e http_proxy="http://172.17.0.1:3128" \
    -e https_proxy="http://172.17.0.1:3128" \
    -e all_proxy="http://172.17.0.1:3128" \
    -it \
    -p 6699:22 \
    --name=spacedrive_local \
    spacedrive:v1.0

