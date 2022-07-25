from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import os
import numpy as np
import math
from dataloaders.rawaudio_util import *


class Base_DataLoader(Dataset):
    """Base Dataset loader."""
    def __init__(self):
        pass

    def retokenized(self, sent, tokenizer):
        tokens = tokenizer.tokenize(sent)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        words = tokenizer.convert_ids_to_tokens(token_ids)
        # if words[-1] != '.':
        #     words.append('.')
        new_sent=' '.join(words)
        return new_sent

    def video_ids_filter(self, video_ids):
        
        video_ids_new=[]
        # vid_not_exist = 0
        # self.audio_fns = os.listdir(self.audio_path)
        for idx,video_id in enumerate(video_ids):
            
            if 'raw_frames' in self.features_path:
                video_path = os.path.join(self.features_path, video_id)
            else:
                video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

            if os.path.exists(video_path) is False:
                continue
            audio_path = os.path.join(self.audio_path, video_id+'.wav')
            if self.filter_video_id == True and os.path.exists(audio_path) is False:
                continue
            video_ids_new.append(video_id)
        # print('Video not exists: {}'.format(vid_not_exist))
        # print('Total videos: {}'.format(len(video_ids_new)))
        return video_ids_new

    def _get_rawvideo(self, video_id, s=[None], e=[None]):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [self.max_frames] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        
        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]

        try:
            for i in range(len(s)):
                start = None if s[i] is None else math.floor(s[i])
                end = None if e[i] is None else math.ceil(e[i])
                if 'raw_frames' in self.features_path:
                    frame_dir = os.path.join(self.features_path, video_id)
                    raw_video_data = self.rawVideoExtractor.get_video_frames(frame_dir, start_time=start, end_time=end, max_frames=self.max_frames, frame_pos=self.slice_framepos)
                else:
                    video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
                    raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time=start, end_time=end)
                
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    
                    if self.max_frames < video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = video_slice[sample_indx, ...]
                    

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] < slice_len else slice_len
                    
                    video[i][:slice_len, ...] = video_slice

        except Exception as e:
            print("video_id: {} error:{}".format(video_id, e))
           
            return video, video_mask

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        
        return video, video_mask

   
    def _get_rawaudio_frames(self, video_id, s=[None], e=[None]):
        audio_mask = np.zeros((len(s), self.max_audio_length), dtype=np.long)
        audio = np.zeros((len(s), self.max_audio_length, 3,
                          self.audio_resolution, self.audio_resolution), dtype=np.float)
        max_audio_length = [0] * len(s)
    
        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]
        audio_path = os.path.join(self.audio_path, video_id+'.wav')
        if not os.path.exists(audio_path):
            
            audio_path = os.path.join(self.audio_path.replace('audios_16k','audios_complement_16k/v2v'),  video_id+'.wav')
           
        try:
            for i in range(len(s)):   
                start = None if s[i] is None else math.floor(s[i])
                end = None if e[i] is None else math.ceil(e[i])

                audio_wav =  get_raw_audio(audio_path, self.audio_rate, start=start, end=end)
                
                audio_frame_l = wav2fbank(audio_wav[:,0], self.audio_rate)
                audio_frame_r = wav2fbank(audio_wav[:,1], self.audio_rate)
                audio_frame_m = wav2fbank((audio_wav[:, 0] + audio_wav[:, 1])/2, self.audio_rate)
                audio_frame =  np.stack([audio_frame_l, audio_frame_m, audio_frame_r], axis=0)
                                
                audio_frame = split_frame(audio_frame, overlap = self.audio_overlap, single_frame_len=self.audio_resolution)
                audio_frame = audio_processor(audio_frame)

                #[tokens_num, channel,tokenlen]
                if self.max_audio_length < audio_frame.shape[0]:
                    # the way deal with long audio keeps the same with video
                    if self.slice_framepos == 0:
                        audio_frame = audio_frame[:self.max_audio_length, ...]
                    elif self.slice_framepos == 1:
                        audio_frame = audio_frame[-self.max_audio_length:, ...]
                    else:
                        start = int((audio_frame.shape[0] - self.max_audio_length)/2)
                        end = start + self.max_audio_length
                        audio_frame = audio_frame[start:end, ...]
                
                slice_len = audio_frame.shape[0]
                max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_len else slice_len
                if slice_len < 1:   
                    pass
                else:
                    audio[i][:max_audio_length[i]] = audio_frame[:max_audio_length[i]]
                    audio_mask[i][:max_audio_length[i]] = [1] * max_audio_length[i]
                    
                    
        except Exception as e:
            '''
            return blank if there exists no audio
            '''
            print("audio_path: {} error:{}".format(audio_path,e))
            
            return audio, audio_mask
        
        return audio, audio_mask

