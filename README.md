# MCLIP4VLA
Mluti-modal multi-lingual Pre-trained model based on CLIP \
This pre-trained model supports 
> English (en) \
> German (de) \
> French (fr) \
> Czech (cs) \
> Chinese (zh) \
> Russian (ru) \
> Vietnamese (vi) \
> Swahili (sw) \
> Spanish (es)
## Setup
```
conda create -n mclip4vla python=3.7
conda activate mclip4vla
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm transformers soundfile opencv-python boto3 ftfy pandas
pip install h5py librosa dominate
```
## Preparation
#### Dataset
Download MSR-VTT from [Baiduyun](https://pan.baidu.com/s/11VWH8VqczIj42LXJ3Y-wkA?pwd=qhq7) (passward:qhq7)
unzip it with```tar -zxvf msrvtt.tar.gz``` and place it in `./data`.
process the dataset with the following command:
```
python data_processor.py --extract_audios --load_video_into_frames
cd data/msrvtt
mv softlink.sh audios_16k/
cd audios_16k
bash softlink.sh
``` 

#### Pre-trained models
Download the pre-trained model from [Baiduyun](https://pan.baidu.com/s/1mISSzAfbCUvLIQHqxH0K9A?pwd=tp1k) (passward:tp1k)

Place it in `./weights`

#### Multilingual tokenizer
Download the multilingual tokenizer setting from  [Baiduyun](https://pan.baidu.com/s/1r4yfR96IGSjYh7ZDx8-N_g?pwd=vth7) (passward:vth7)

Unzip it with `tar -zxvf M-BERT-Based-69-ViT-B.tar.gz` and place it in `./weights`.

## Finetuning 
To evaluate our model, run
```
bash msrvtt_finetune.sh
```
the results should be 

| Method | en   | zh   | cs   | de   | es   | fr   | ru   | sw   | vi   | avg  |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Ours   | 35.8 | 30.4 | 32.7 | 33.7 | 33.5 | 33.8 | 32.0 | 22.6 | 15.9 | 30.0 |




