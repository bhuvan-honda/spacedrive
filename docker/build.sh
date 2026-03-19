#!/bin/bash

set -ex

# proxy to be used inside the docker, leave blank if you are not using
# a proxy server
: "${INSIDE_DOCKER_PROXY:=http://172.17.0.1:3128}"

# we set the docker name to the project name and use the tag to indicate
# the version/env 
: "${PROJECT_NAME:=spacedrive}"
: "${DOCKER_TAG:=v1.0}"
# # # # # # # # # # #
# BUILD
# # # # # # # # # # #

# actual build
sudo docker build . \
  --build-arg INSIDE_DOCKER_PROXY=$INSIDE_DOCKER_PROXY \
  --build-arg HOST_UNAME=$USER \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) \
  -t $PROJECT_NAME:$DOCKER_TAG
