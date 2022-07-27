# MCLIP4VLA
Mluti-modal multi-lingual Pre-trained model

## Setup
```
conda create -n mclip4vla python=3.7
conda activate mclip4vla
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm transformers soundfile opencv-python boto3 ftfy
```
## Dataset 
Download MSR-VTT from [Baiduyun](https://pan.baidu.com/s/1mISSzAfbCUvLIQHqxH0K9A?pwd=tp1k) (passward:tp1k)
place it in `./data` and unzip it with```tar -zxvf msrvtt.tar.gz```
process the dataset with the following command:
```
mkdir data
python data_processor --extract_audios --load_video_into_frames
cd data/msrvtt
bash audio_softlink.sh
``` 

## Pre-trained models
Download the pre-trained model from [Baiduyun]() passward:
## Quick Start

## Finetuning 

## License



