#!/bin/bash

# DATADIR=<path_to_your_data_directory>
DATADIR="data/nuscenes"
# WORKDIR=<path_to_your_code_directory>
WORKDIR="/home/J0986625/SpaceDrive/"
# CKPTDIR=<path_to_your_checkpoint_directory>
CKPTDIR="ckpts"

PROJECT_NAME="spacedrive"
# flash-attn container: DOCKER_TAG=v1.0-flashattn CONTAINER_NAME=spacedrive_flash SSH_PORT=6700 ./deploy_local_docker.sh
: "${DOCKER_TAG:=v1.0}"
: "${CONTAINER_NAME:=spacedrive_local}"
: "${SSH_PORT:=6699}"

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
    -p $SSH_PORT:22 \
    --name=$CONTAINER_NAME \
    $PROJECT_NAME:$DOCKER_TAG

