# Data & Model Preperation 
## Prepare Model
The current version of SpaceDrive supports the following base VLMs:
- [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl)
    - [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
    - [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [llava-1.5](https://huggingface.co/collections/llava-hf/llava-15)
    - [llava-1.5-7b](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

### 1. Download model

Save the downloaded model checkpoints under `./ckpts`.

### 2. Add extra tokens to the base VLM

In order to add <POS_INDICATOR> and <POS_EMBEDDING> tokens to the VLM, run:
```shell
# NOTE: remeber to change ckpt_path in token_additor.py
python tools/token_additor.py
```
Adapt the script parameters, e.g., the path to model checkpoint, as needed.

### 3. Change Token ID 
Based on the chosen VLM, change the corresponding token id in [constants.py](../projects/mmdet3d_plugin/datasets/utils/constants.py)



## Prepare Dataset

### 1. Download

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

Follow the instructions in [nuScenes-OmniDrive](https://github.com/NVlabs/OmniDrive/blob/main/docs/setup.md) to download info files and placed them under `./data/nuscenes`. 

### 2. Add command descriptions
```bash
python tools/command_generation.py
```