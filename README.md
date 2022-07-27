# MCLIP4VLA
Mluti-modal multi-lingual Pre-trained model

## Setup
```
conda create -n mclip4vla python=3.7
conda activate mclip4vla
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm transformers soundfile python-cv
```
## Dataset 
Download MSR-VTT from [Baiduyun]() passward:
place it in `./data` and unzip it
process the dataset with the following command:
```
python data_processor --extract_audios --load_video_into_frames
cd dataset/msrvtt
bash audio_softlink.sh

``` 
## Quick Start
## Finetuning 

## License



