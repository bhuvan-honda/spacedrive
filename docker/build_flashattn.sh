#!/bin/bash
# Builds spacedrive:v1.0-flashattn on top of spacedrive:v1.0 (build that first with build.sh).

set -ex

: "${PROJECT_NAME:=spacedrive}"
: "${BASE_TAG:=v1.0}"
: "${DOCKER_TAG:=${BASE_TAG}-flashattn}"

docker build . \
  -f Dockerfile.flashattn \
  --build-arg BASE_IMAGE=$PROJECT_NAME:$BASE_TAG \
  -t $PROJECT_NAME:$DOCKER_TAG
