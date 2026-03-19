# Environment Setup

We provide two ways to create the environment for SpaceDrive. **However, we strongly recommand to use the docker setup.**

## Docker Setup

First, if needed, modifify the proxy used inside the docker in both `docker/build.sh` and `docker/deploy_local_docker.sh`. 

In `docker/deploy_local_docker.sh`, enter the dataset path and codebase path.

Then run:

```shell
cd docker

# create docker image
bash ./build.sh

# deploy your created image
bash ./deploy_local_docker.sh
```

## Conda Setup

```shell
conda create -n spacedrive python=3.9
conda activate spacedrive

pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir transformers==4.52.1 peft==0.14.0 accelerate==1.7.0
pip install --no-cache-dir qwen-vl-utils[decord]==0.0.8

pip install --no-cache-dir mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install --no-cache-dir mmdet==2.28.2
pip install --no-cache-dir mmsegmentation==0.30.0

cd mmdetection3d
git checkout v1.0.0rc6 
pip install --no-build-isolation -e .
cd ..

cd unidepth
pip install -e .
cd ..

pip install --no-cache-dir -r docker/requirements.txt

sudo cp -rf docker/torch21_mmcv/_functions.py $CONDA_PREFIX/lib/python3.9/site-packages/mmcv/parallel/_functions.py
sudo cp -rf docker/torch21_mmcv/distributed.py $CONDA_PREFIX/lib/python3.9/site-packages/mmcv/parallel/distributed.py
sudo cp -rf docker/torch21_mmcv/__init__.py ./mmdetection3d/mmdet3d/__init__.py
```