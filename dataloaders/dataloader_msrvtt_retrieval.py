from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import pickle
import pandas as pd
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.dataloader_base import Base_DataLoader
from dataloaders.rawaudio_util import *


class MSRVTT_Retrieval_DataLoader(Base_DataLoader):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            json_path,
            frame_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            
            frame_order=0,
            slice_framepos=0,
            audio_path=None,
            max_audio_length=12,
            audio_resolution=224,
            audio_tokenlen=1,  
            audio_channel=2,
            audio_rate=16000, 
            audio_overlap=0,
             
            video_path = None,
    ):
        super(MSRVTT_Retrieval_DataLoader, self).__init__()
        self.data = json.load(open(json_path, 'r'))
        
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_path = frame_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)


        self.audio_path = audio_path
        self.max_audio_length = max_audio_length
        self.audio_overlap = audio_overlap
        self.audio_tokenlen = audio_tokenlen
        self.audio_channel = audio_channel
        self.audio_rate = audio_rate
        self.audio_resolution = audio_resolution
        self.video_path = video_path

        self.feature_dict = {}
        if os.path.isfile(self.frame_path):
            self.feature_dict = pickle.load(open(self.frame_path, 'rb'))
            self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]
        val_video_ids = [i[:-4] for i in self.data.keys()]
        val_video_ids = self.video_ids_filter(val_video_ids)
        
        
        self.video_id2idx=dict(zip(val_video_ids, list(range(len(val_video_ids)))))
        self.sentences_dict={}
        for video_id in val_video_ids:
            self.sentences_dict[len(self.sentences_dict)] = (video_id, self.data[video_id+'.mp4'][0])
    
    def __len__(self):
        return len(self.sentences_dict)
        
    def get_meta(self):
        meta = {"video_path":[],"captions":[]}
        for video_id, sentence in self.sentences_dict.values():
            meta["video_path"].append(os.path.join(self.video_path, video_id +'.mp4'))
            meta["captions"].append(sentence)
        return meta

    def _get_text(self, caption):
        input_ids = np.zeros(self.max_words, dtype=np.long)
        attention_mask = np.zeros(self.max_words, dtype=np.long)
        
        txt_dict = self.tokenizer(caption, padding=True, max_length=self.max_words, return_tensors='np', truncation=True)
        txt_id = txt_dict['input_ids']
        txt_mask =txt_dict['attention_mask']
        if txt_dict['input_ids'].shape[0] == 1:
            txt_id = txt_id.squeeze(0)
            txt_mask = txt_mask.squeeze(0)
        txt_len = len(txt_id)
        input_ids[:txt_len]= txt_id
        attention_mask[:txt_len] = txt_mask
        
        return input_ids, attention_mask

    def __getitem__(self, idx):
        video_id, sentence = self.sentences_dict[idx]
    
        txt_id, txt_mask = self._get_text(sentence)
        video, video_mask  = self._get_rawvideo(video_id)
        audio, audio_mask  = self._get_rawaudio_frames(video_id)
        
        return txt_id, txt_mask, video, video_mask, audio, audio_mask



class MSRVTT_Retrieval_TrainDataLoader(Base_DataLoader):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            frame_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            unfold_sentences=False,
            
            audio_path=None,
            audio_overlap=0,
            max_audio_length=12,
            audio_resolution=224, 
            audio_tokenlen=1,
            audio_rate=16000,
            audio_channel=2,

    ):
        super(MSRVTT_Retrieval_TrainDataLoader, self).__init__()
        self.csv = pd.read_csv(csv_path)
        
        self.data = json.load(open(json_path, 'r'))
        
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_path = frame_path


        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.audio_path = audio_path      
        self.max_audio_length = max_audio_length
        
        self.audio_overlap = audio_overlap
        self.audio_resolution = audio_resolution
        self.audio_rate = audio_rate
        self.audio_channel = audio_channel
        self.audio_tokenlen = audio_tokenlen
       
        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        self.feature_dict = {}
      
        
        train_video_ids = list(self.csv['video_id'].values)
        train_video_ids = self.video_ids_filter(train_video_ids)

        self.video_id2idx=dict(zip(train_video_ids, list(range(len(train_video_ids)))))
        sentences_dict = {}
        for video_id in train_video_ids:
            sentences = self.data[video_id+'.mp4']
            if 'captions_all' in json_path:
                sentences=random.sample(sentences, 20)
            for sentence in sentences: 
                sentences_dict[len(sentences_dict)] = (video_id, sentence)


        self.sentences_dict = sentences_dict
        self.sample_len = len(self.sentences_dict)
        

    def __len__(self):
        return self.sample_len

    def _get_text(self, caption):
        
        input_ids = np.zeros(self.max_words, dtype=np.long)
        attention_mask = np.zeros(self.max_words, dtype=np.long)
        
        txt_dict = self.tokenizer(caption, padding=True, max_length=self.max_words, return_tensors='np', truncation=True)
        
        txt_id = txt_dict['input_ids']
        txt_mask =txt_dict['attention_mask']
        if txt_dict['input_ids'].shape[0] == 1:
            txt_id = txt_id.squeeze(0)
            txt_mask = txt_mask.squeeze(0)
        txt_len = len(txt_id)
        input_ids[:txt_len]= txt_id
        attention_mask[:txt_len] = txt_mask
        
        return input_ids, attention_mask

    def __getitem__(self, idx):
        
        video_id, caption = self.sentences_dict[idx]
        txt_id, txt_mask = self._get_text(caption)
        video, video_mask = self._get_rawvideo(video_id)
        audio, audio_mask  = self._get_rawaudio_frames(video_id)
        return txt_id, txt_mask, video, video_mask, audio, audio_mask
