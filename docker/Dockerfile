FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# To solve arch_list[-1] += '+PTX'
ARG TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"

ARG INSIDE_DOCKER_PROXY
ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID

ARG DEBIAN_FRONTEND=noninteractive

ENV http_proxy="${INSIDE_DOCKER_PROXY}"
ENV https_proxy="${INSIDE_DOCKER_PROXY}"
ENV HTTP_PROXY="${INSIDE_DOCKER_PROXY}"
ENV HTTPS_PROXY="${INSIDE_DOCKER_PROXY}"
ENV all_proxy="${INSIDE_DOCKER_PROXY}"

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

SHELL ["/bin/bash", "-c"]
RUN set -ex && \
    apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential libfreetype6-dev libhdf5-serial-dev libpng-dev libzmq3-dev pkg-config software-properties-common unzip sudo libtiff-dev iproute2 \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
    git \
    less nano vim \
    rsync liblz4-dev \
    ca-certificates libcurl3-gnutls curl wget \
    cmake make clang-6.0 unixodbc-dev apt-utils apt-transport-https gnupg \
    libgl1-mesa-dev libstdc++6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Install pytorch
RUN pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
RUN pip install transformers==4.52.1 peft==0.14.0 accelerate==1.7.0 \
    && pip install qwen-vl-utils[decord]==0.0.8

# Install OpenMMLab packages
ENV MMCV="1.7.2"
ENV MMDET="2.28.2"
ENV MMSEG="0.30.0"
ENV MMDET3D="1.0.0rc6"

RUN mkdir /deps

RUN TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6;9.0+PTX" pip install mmcv-full==${MMCV} \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

RUN pip install mmdet==${MMDET}

RUN pip install mmsegmentation==${MMSEG}

RUN cd /deps \
    && git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout tags/v${MMDET3D} \
    && pip install "numpy<2.0.0" "scikit-image<0.20" "scikit-learn<1.3" "pandas<2.0.0" "opencv-python<4.9.0" "opencv-python-headless<4.9.0" "absl-py<2.0.0"\
    && python3.9 setup.py develop

# Install unidepth_py3.9
RUN git clone https://github.com/zhenghao2519/UniDepth_py39 \
    && cd UniDepth_py39 \
    && pip install -e .

# Install other packages
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Replace for torch 21 device type
COPY ./torch21_mmcv/_functions.py /tmp/_functions.py
COPY ./torch21_mmcv/distributed.py /tmp/distributed.py
COPY ./torch21_mmcv/__init__.py /tmp/__init__.py
RUN cp -rf /tmp/_functions.py /usr/local/lib/python3.9/dist-packages/mmcv/parallel/_functions.py \
    && cp -rf /tmp/distributed.py /usr/local/lib/python3.9/dist-packages/mmcv/parallel/distributed.py \
    && cp -rf /tmp/__init__.py /deps/mmdetection3d/mmdet3d/__init__.py

# Install ssh
RUN apt update \
    && echo Y | apt install openssl openssh-server openssh-client

WORKDIR /workspace

# Add user
RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
RUN useradd --no-log-init -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME} \
    && echo ${HOST_UNAME}:${HOST_UNAME} | chpasswd

USER root
RUN usermod -aG sudo ${HOST_UNAME}
RUN echo "${HOST_UNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY ./inputrc /root/.inputrc
RUN cp /root/.inputrc /home/${HOST_UNAME}/.inputrc \
    && chown ${HOST_UNAME}:${HOST_GID} /home/${HOST_UNAME}/.inputrc

WORKDIR /workspace