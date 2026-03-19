FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# To solve arch_list[-1] += '+PTX'
ARG TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"

# proxy needs to be provided from the outside
ARG INSIDE_DOCKER_PROXY
ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID

# need this for apt to not get stuck prompting the user for locale info
ARG DEBIAN_FRONTEND=noninteractive

# set proxy
ENV http_proxy="${INSIDE_DOCKER_PROXY}"
ENV https_proxy="${INSIDE_DOCKER_PROXY}"
ENV HTTP_PROXY="${INSIDE_DOCKER_PROXY}"
ENV HTTPS_PROXY="${INSIDE_DOCKER_PROXY}"
ENV all_proxy="${INSIDE_DOCKER_PROXY}"

# Arguments to build Docker Image using CUDA
ARG USE_CUDA
ARG TORCH_ARCH

ENV AM_I_DOCKER=True
ENV FORCE_CUDA="1"
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install essentials (debian packages for Ubuntu 18.04)
SHELL ["/bin/bash", "-c"]
RUN set -ex && \
    # install debian packages
    apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential libfreetype6-dev libhdf5-serial-dev libpng-dev libzmq3-dev pkg-config software-properties-common unzip sudo libtiff-dev iproute2 \
    # python
    python3 python3-pip \
    # nice to have: git
    git \
    # nice to have: common editors
    less nano vim \
    # nice to have: copy files around
    rsync liblz4-dev \
    # nice to have: internet access
    ca-certificates libcurl3-gnutls curl wget \
    # build tools
    cmake make clang-6.0 unixodbc-dev apt-utils apt-transport-https gnupg \
    # nice to have: libGL.so.1 required by python package cv2
    libgl1-mesa-dev libstdc++6 \
    # apt-get clean
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && conda update conda \
    && conda install -y python=3.9 pip

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && conda init bash \
    && conda config --set auto_activate_base true

# Install pytorch
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# support for qwen2.5_vl starts from transformers 4.49.0
# Use direct wheel URLs to bypass the proxy's stale PyPI simple index cache
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && pip install transformers==4.52.1 peft==0.14.0 accelerate==1.7.0 \
    && pip install qwen-vl-utils[decord]==0.0.8

# Install OpenMMLab packages
ENV MMCV="1.7.2"
ENV MMDET="2.28.2"
ENV MMSEG="0.30.0"
ENV MMDET3D="1.0.0rc6"

# RUN nvidia-smi
RUN mkdir /deps
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6;9.0+PTX" pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && pip install mmdet==${MMDET}

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && pip install mmsegmentation==${MMSEG}

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && cd /deps \
    && git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout tags/v${MMDET3D} \
    && pip install --no-build-isolation -v -e .

# install unidepth_py3.9
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && git clone https://github.com/zhenghao2519/UniDepth_py39 \
    && cd UniDepth_py39 \
    && pip install -e .

# Install other packages
COPY ./requirements.txt /tmp/requirements.txt
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && pip install -r /tmp/requirements.txt

# Replace for torch 21 device type
COPY ./torch21_mmcv/_functions.py /tmp/_functions.py
COPY ./torch21_mmcv/distributed.py /tmp/distributed.py
COPY ./torch21_mmcv/__init__.py /tmp/__init__.py
RUN cp -rf /tmp/_functions.py /opt/conda/lib/python3.9/site-packages/mmcv/parallel/_functions.py \
    && cp -rf /tmp/distributed.py /opt/conda/lib/python3.9/site-packages/mmcv/parallel/distributed.py \
    && cp -rf /tmp/__init__.py /deps/mmdetection3d/mmdet3d/__init__.py

# Install ssh
RUN apt update \
    && echo Y | apt install  openssl openssh-server openssh-client

WORKDIR /workspace

# add user of current host sytem
RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
RUN useradd --no-log-init -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME} \
    && echo ${HOST_UNAME}:${HOST_UNAME} | chpasswd
USER ${HOST_UNAME}

# make sure conda env is always active on startup there as well
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && conda init bash \
    && conda config --set auto_activate_base true

# switch back to root as default user
USER root
RUN usermod -aG sudo ${HOST_UNAME}
RUN echo "${HOST_UNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# copy inputrc to user for a nicer bash experience
COPY ./inputrc /root/.inputrc
RUN cp /root/.inputrc /home/${HOST_UNAME}/.inputrc \
    && chown ${HOST_UNAME}:${HOST_GID} /home/${HOST_UNAME}/.inputrc

WORKDIR /workspace


 