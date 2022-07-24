# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import torch, pdb
import time
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from util import mylog
from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, HardCrossEn, MILNCELoss, MaxMarginRankingLoss, AllGather, Loss_recoder
from modules.module_audio import CLIP4Audio
from modules.module_clip import CLIP, convert_weights, ClipOnlyMLMHead
from modules.module_cross import CrossModel_Clip
from modules.tokenization import END_TOKEN, VOCAB_SIZE
from torch.utils.checkpoint import checkpoint


coffient = 100.0
logger = mylog().getlog()#logging.getLogger(__name__)
allgather = AllGather.apply

def show_log(task_config, info):
    
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

class AudioClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self,*inputs, **kwargs):
        super(AudioClipPreTrainedModel, self).__init__()
        
        self.clip = None
        self.audio = None
        self.cross = None
        self.decoder = None

    @classmethod
    def from_pretrained(cls, state_dict=None, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        audio_state_dict = None 
        pretrained_clip_name = "RN50x16"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)

        
        ###############################################
        pre_trained_model = ''
        if state_dict is None: 
            state_dict = {}
            
            if clip_state_dict is not None:
                pre_trained_model += 'clip/'+ pretrained_clip_name   
        else:
            pre_trained_model = 'initial_model'
  
        if clip_state_dict is not None:
            for key, val in clip_state_dict.items():   
                if key.startswith('transformer'): continue       
                new_key = "clip." + key
                
                if  new_key not in state_dict:
                    state_dict[new_key] = val.clone()       
        
        
        model = cls(clip_state_dict,  *inputs, **kwargs)
        # initialize cross-encoder with text-encoder of clip

        if (check_attr('stage_two', task_config) or check_attr('train_sim_after_cross', task_config) ):
            for name, param in model.cross.named_parameters():
                cross_param_name = "cross." + name
                if  name == 'cross_projection':
                    name = 'text_projection'
                clip_param_name = name
                if cross_param_name not in state_dict.keys() and clip_param_name in clip_state_dict.keys():
                    state_dict[cross_param_name] = clip_state_dict[clip_param_name].clone()
                    
        # initialize audio backbone with image-encoder of clip
        if (check_attr('no_audio_initialize', task_config) is False):
             for name, param in model.audio.named_parameters():
                audio_param_name = "audio." + name
                clip_param_name = "clip." + name
                
                if audio_param_name in state_dict.keys():
                    continue
                if  clip_param_name in state_dict.keys():
                    state_dict[audio_param_name] = state_dict[clip_param_name].clone()
                    
        # assert model.audio is not None
        assert model.clip is not None

        model = cls.init_preweight(model, state_dict, task_config=task_config, pre_trained_model=pre_trained_model)
        

        ## ####################################
        # freeze layers
        ## ####################################
        
        assert task_config.freeze_layer_num <= 13 and task_config.freeze_layer_num >= -1
        if task_config.freeze_layer_num == 13:
            show_log(task_config, "Freeze all clip params. ")
        elif task_config.freeze_layer_num == -1:
            show_log(task_config, "Training all clip params. ")
           

        if hasattr(model, "clip") and task_config.freeze_layer_num > -1:
            for name, param in model.clip.named_parameters():  
                if task_config.freeze_layer_num == 13:
                    param.requires_grad= False
                    continue
                
                if task_config.freeze is not None and name.find(task_config.freeze)==0:
                    param.requires_grad= False
                    show_log(task_config, "Freeze Parameter clip.{} ".format(name))
                    continue    # need to train
                # top layers always need to train
                if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                        or name.find("visual.attnpool") == 0 or name.find("visual.proj") == 0:
                    show_log(task_config, "Training Parameter clip.{} ".format(name))
                    continue    # need to train
                elif name.find("visual.transformer.resblocks.") == 0 or  name.find("visual.layer") == 0 or name.find("transformer.resblocks") == 0:
                    if name.find("visual.layer") == 0:
                        layer_num = int(name.split("visual.layer")[1].split(".")[0])
                    else:
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= task_config.freeze_layer_num:
                        show_log(task_config, "Training Parameter clip.{}  ".format(name))
                        continue    # need to train

                
                    # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False
                show_log(task_config, "Freezed Parameter clip.{} ".format(name))

        if task_config.freeze is not None:
             for name, param in model.named_parameters():
                 if name.find(task_config.freeze)==0:
                      param.requires_grad= False

        num_params_total = sum(p.numel() for p in model.parameters())
        num_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            params_total_label = 'k'
        show_log(task_config,"Total Parameters:{:.2f}{}".format(num_params_total,params_total_label))
        show_log(task_config,"Total Training Parameters:{:.2f}{}".format(num_params_train,params_total_label))     

        return model



def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class AudioClip(AudioClipPreTrainedModel):
    
    def __init__(self,  clip_state_dict, task_config):
        super(AudioClip, self).__init__( clip_state_dict)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.ignore_audio_index = -1
        self.ignore_index = -2

       
        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))
        
        self.train_sim_after_cross = False
        if  check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")
        
      
         # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = counts #len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".ln_2.weight")])
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        self.vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        
        transformer_heads = transformer_width // 64
        # transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(self.vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        # show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))
        show_log(self.task_config,"\t loss_type:{}".format(self.task_config.loss_func))

      
        self.gradient_checkpoint = False
        if check_attr("gradient_checkpoint", self.task_config):
            self.gradient_checkpoint = True
            show_log(task_config, "\t gradient_checkpoint: {}".format(self.gradient_checkpoint))
        

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, self.vocab_size, transformer_width,
            
        ).float()
        
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        if self.task_config.fp16 == False:
            convert_weights(self.clip.visual)
        # <=== End of CLIP Encoders
        #word_embeddings_weight = self.clip.token_embedding.weight
        #position_embeddings_weight = self.clip.positional_embedding

        # Audio Encoder ===>
     
        self.audio = CLIP4Audio(embed_dim=embed_dim,
        image_resolution = image_resolution, vision_layers = vision_layers, \
            vision_width = vision_width, vision_patch_size = vision_patch_size, \
                with_bg_token=self.task_config.with_bg_token, with_control_token=self.task_config.with_control_token, \
                is_pretrain = self.task_config.do_pretrain).float()
        if self.task_config.fp16 == False:
            convert_weights(self.audio)

        # <=== End of Audio Encoder
          
        if self.train_sim_after_cross or self.task_config.task_type=='caption':
            # Cross Encoder ===>
            cross_num_hidden_layers = self.task_config.cross_num_hidden_layers
            self.cross = CrossModel_Clip(max_position_embeddings = context_length,
                hidden_size = embed_dim,
                type_vocab_size = 3,
                num_hidden_layers = cross_num_hidden_layers,
                num_attention_heads = transformer_heads,
                hidden_dropout_prob = 0.1)
            
            # <=== End of Cross Encoder      

            
            # finetune stage && without decoder
            
        # pre_p2 or finetune with_decoder
        if self.task_config.task_type == 'caption':
            self.cls = ClipOnlyMLMHead(embed_dim, word_embeddings_weight)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        if self.task_config.train_sim_after_cross == True and self.task_config.task_type == 'retrieval':   
            self.similarity_dense = nn.Linear(embed_dim, 1)

        '''
        mil-NCE loss for joint matching(not cross matching)
        '''

        # weights for classification ===>
        if self.task_config.task_type=='classification' and self.task_config.class_num!=0:
            self.cls_linear = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, self.task_config.class_num))
            if self.task_config.datatype=='esc50':
                self.cls_loss = CrossEntropyLoss()
                
            elif self.task_config.datatype=='audioset':
                self.cls_loss = BCEWithLogitsLoss()
        #<===  weights for classification

        self.nceloss = CrossEn() 


        '''init sub layers of self.init_weights and tied to self'''
        if check_attr('with_self_supervised',self.task_config):
            self.simclr_loss = CrossEntropyLoss()
        self.apply(self.init_weights)

    def forward(self, input_txt=None, attention_mask=None, video=None, video_mask=None,
                audio=None, audio_mask=None, input_ids=None, bg_token_gt=None, lang='en'):
        
        if input_txt is not None:
            
            sequence_output = self.get_sequence_output(input_txt,attention_mask, lang=lang)
        else:
            sequence_output = None
            attention_mask = None
    
                                                                                                      
        # attention_mask: text token mask
        if video is not None and video_mask is not None:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
            visual_output = self.get_visual_output(video, video_mask,  shaped=True, video_frame=video_frame)
            
        else:
            visual_output = None
            video_mask = None
      
        if audio is not None and audio_mask is not None:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            audio = torch.as_tensor(audio).float()
            b, pair, bs  = audio.shape[:3]
            audio = audio.view(b * pair * bs , *audio.shape[3:]) 
            audio_output, bg_tokens = self.get_audio_output(audio, audio_mask, token_type=bg_token_gt, shaped=True)
        else:
            audio_output = None
            audio_mask = None
        
   
        
        '''
        sequence_output:[batchsize, max_word_len, 512]
        visual_output:[batchsize, max_frame_len, 512]
        audio_output:[batchsize, max_audio_len, 512]
        '''
        if self.training:
            loss_recoder = Loss_recoder()
            loss = 0.
            
            if self._stage_one:   
                #sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]    
             
                if visual_output is None and video_mask is None:  
                    sim_matrix_t_a = self.get_similarity_logits(sequence_output, audio_output, attention_mask, audio_mask,'text','audio',input_ids=input_ids, shaped=True)     
                
                    sim_loss_t_a = self.nceloss(sim_matrix_t_a)
                
                    loss_recoder.update('ta_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_a)
                    loss+=sim_loss_t_a

                elif audio_output is None and audio_mask is None:
                    sim_matrix_t_v = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,'text','video', input_ids=input_ids,shaped=True)
                        
                    sim_loss_t_v = self.nceloss(sim_matrix_t_v) 
                    loss_recoder.update('tv_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_v)
                    loss+=sim_loss_t_v

                elif self.task_config.retrieval_finetune=='tight_seq' or 'cross_align':
                    sim_matrix = self.get_similarity_logits_with_three_modalities(sequence_output, visual_output, audio_output, \
                    attention_mask, video_mask, audio_mask,input_ids=input_ids, shaped=True)['t_va']
                    sim_loss_t_av_nce= self.nceloss(sim_matrix)
                        
                    loss_recoder.update('tav_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_av_nce)
                    loss += sim_loss_t_av_nce

                else:
                    sim_dict= self.get_similarity_logits_with_three_modalities(sequence_output, visual_output, audio_output, \
                        attention_mask, video_mask, audio_mask,input_ids=input_ids, shaped=True)
                        
                      
                    if  'nce' in self.task_config.loss_func:
                        sim_loss_t_v_nce = self.nceloss(sim_dict['t_v']) 
                        sim_loss_t_a_nce = self.nceloss(sim_dict['t_a']) 
                        sim_loss_v_a_nce = (self.nceloss(sim_dict['a_v']) + self.nceloss(sim_dict['a_v'].T)) / 2 
                        sim_loss_t_av_nce = self.nceloss(sim_dict['t_va'])
                            
                        loss_recoder.update('va_nce', sim_loss_v_a_nce)
                        loss_recoder.update('ta_nce', sim_loss_t_a_nce)
                        loss_recoder.update('tv_nce', sim_loss_t_v_nce)
                        loss_recoder.update('tav_nce', sim_loss_t_av_nce)
                        if self.task_config.loss_func == 'ta_nce':
                            loss += sim_loss_t_a_nce 
                        elif self.task_config.loss_func == 'tv_nce':
                            loss += sim_loss_t_v_nce 
                        elif self.task_config.loss_func == 'av_nce':
                            loss += sim_loss_v_a_nce
                        elif self.task_config.loss_func == 'tav_nce':
                            loss += sim_loss_t_av_nce
                        else:
                            raise NotImplementedError

            if self._stage_two:
             
                if self.task_config.task_type == "retrieval":
                   
                   

                    input_dict={'sequence_output':sequence_output,'input_ids':input_ids, 'shaped':True}
                        
                    if visual_output ==  None and video_mask == None:
                        input_dict.update({'audio_output':audio_output,'audio_mask':audio_mask})
                        loss_type='ta'
                    elif audio_output == None and audio_mask == None:
                        input_dict.update({'visual_output':visual_output,'video_mask':video_mask})
                        loss_type='tv'
                    else: 
                        input_dict.update({'visual_output':visual_output, \
                            'video_mask': video_mask,'audio_output':audio_output, 'audio_mask': audio_mask})
                        loss_type='tav'
                        
                    if loss_type == 'tv':
                        input_dict.update({'modal1':'text', 'modal2':'video'})
                        input_dict['modal1_output']= input_dict.pop('sequence_output')
                        input_dict['modal1_mask']= input_dict.pop('modal1_mask')
                        input_dict['modal2_output']= input_dict.pop('visual_output')
                        input_dict['modal2_mask']= input_dict.pop('video_mask')
                        sim_matrix = self.get_similarity_logits(**input_dict)
                                 
                    elif loss_type == 'ta':
                        input_dict['modal1_output']= input_dict.pop('sequence_output')
                        input_dict['modal2_output']= input_dict.pop('audio_output')
                        input_dict['modal2_mask']= input_dict.pop('audio_mask')
                        input_dict.update({'modal1':'text', 'modal2':'audio'})
                        sim_matrix = self.get_similarity_logits(**input_dict)
                               
                    elif loss_type == 'tav':
                        
                        sim_matrix = self.get_similarity_logits_with_three_modalities(**input_dict)['t_va']
                    
                    
                    sim_loss= self.nceloss(sim_matrix)
                    loss_recoder.update('align_nce_'+loss_type, sim_loss)
                    loss += sim_loss
                else:
                    raise NotImplementedError        

            return loss, loss_recoder
        else:
            return None

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        
        if self.task_config.pretrained_clip_name.startswith('RN'):
            visual_hidden = self.clip.encode_image_resnet(video).float()
        else:
            visual_hidden = self.clip.encode_image_transformer(video).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        
        return visual_hidden

    def get_sequence_output(self, input_txt, attention_mask):
        sequence_hidden= self.clip.encode_text(input_txt, attention_mask)
        return sequence_hidden
    
    def get_audio_output(self, audio, audio_mask, token_type=None,shaped=False, with_hidden=False):
        if shaped is False:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            
            b, pair, ts = audio.shape[:3]
            audio = audio.view(b * pair*ts , *audio.shape[3:])

        if token_type is not None:
            token_type = token_type.expand(audio_mask.shape).reshape(-1).to(torch.int64)    

        bs_pair = audio_mask.size(0)
        audio_hidden, bg_hidden  = self.audio(audio, token_type = token_type)
        audio_hidden = audio_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        if bg_hidden is not None:
            bg_hidden = bg_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        
        return audio_hidden, bg_hidden
               
    def _get_cross_output(self, sequence_output=None, visual_output=None, audio_output=None, attention_mask=None, video_mask=None, audio_mask=None, concat_mask=None, input_ids=None):
        # concatnate tokens and frames 
       
        if audio_output is None and audio_mask is  None: 
            text_type_ = torch.zeros(attention_mask.shape[:2], dtype=torch.long, device=attention_mask.device) 
            video_type_ = torch.ones(video_mask.shape[:2], dtype=torch.long, device=video_mask.device)
            concat_type = torch.cat((text_type_,  video_type_), dim=1)
            concat_features = torch.cat((sequence_output,  visual_output), dim=1)
            if concat_mask is None: concat_mask = torch.cat((attention_mask,  video_mask), dim=1)
        elif visual_output is None and video_mask is None:
            text_type_ = torch.zeros(attention_mask.shape[:2], dtype=torch.long, device=attention_mask.device) 
            audio_type_ = torch.ones(audio_mask.shape[:2], dtype=torch.long, device=audio_mask.device)+1
            
            concat_type = torch.cat((text_type_,  audio_type_), dim=1)
            concat_features = torch.cat((sequence_output,  audio_output), dim=1)
            if concat_mask is None: concat_mask = torch.cat((attention_mask,  audio_mask), dim=1)
        elif sequence_output is  None and attention_mask is None:
            video_type_ = torch.ones(video_mask.shape[:2], dtype=torch.long, device=video_mask.device)
            audio_type_ = torch.ones(audio_mask.shape[:2], dtype=torch.long, device=audio_mask.device)+1
            concat_type = torch.cat((video_type_, audio_type_), dim=1)
            concat_features = torch.cat((visual_output, audio_output), dim=1)
            if concat_mask is None: concat_mask = torch.cat((video_mask,  audio_mask), dim=1)
        else:
            
            text_type_ = torch.zeros(attention_mask.shape[:2], dtype=torch.long, device=attention_mask.device) 
            video_type_ = torch.ones(video_mask.shape[:2], dtype=torch.long, device=video_mask.device)
            audio_type_ = torch.ones(audio_mask.shape[:2], dtype=torch.long, device=audio_mask.device)+1

            concat_type = torch.cat((text_type_, video_type_, audio_type_), dim=1)
            concat_features = torch.cat((sequence_output, visual_output, audio_output), dim=1)
            if concat_mask is None: concat_mask = torch.cat((attention_mask, video_mask, audio_mask), dim=1)

        #print("concat_features:{}, concat_type:{}, concat_mask:{}".format(concat_features.mean(dim=[1,2]), \
            # concat_type.sum(dim=1), concat_mask.sum(dim=1)))
        cross_output, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=False, input_ids=input_ids)
        # cross_output = cross_layers[-1]
        # if input_ids is not None:
        #     pooled_output = cross_output[torch.arange(cross_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_single_modal(self, modal_output, modal_mask, modal_type):
        assert modal_type in ['video', 'text', 'audio']
        if modal_type == 'text' :# add an [cls] token
            modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_text_len, 1]
            modal_mask_un[:, 0, :] = 0. #[cls] token
            modal_output = modal_output * modal_mask_un
            modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
            modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
            modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
        #[batchsize, text_dim]
        elif modal_type == 'video' or modal_type == 'audio':
            if modal_output.shape[1] == modal_mask.shape[1]:
                modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_frame_len, 1]
                modal_output = modal_output * modal_mask_un
                modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
                modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
                modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
            else:
                modal_out = modal_output.mean(dim=1)
        #[batchsize, frame_dim]
        
        return modal_out

    def _cross_similarity(self, sequence_output, attention_mask, visual_output=None, audio_output=None, video_mask=None, audio_mask=None, concat_mask=None, input_ids=None):
        # wating to be filled

        if self.training:
            #print('device:{}, batchsize:{}, audio_output[0]:{}'.format(audio_output.device, audio_output.shape[0],  audio_output[:,0,0]))
            
            sequence_output = allgather(sequence_output, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            if visual_output is not None: visual_output = allgather(visual_output, self.task_config)
            if video_mask is not None: video_mask = allgather(video_mask, self.task_config)
            if audio_output is not None: audio_output = allgather(audio_output, self.task_config)
            if audio_mask is not None: audio_mask = allgather(audio_mask, self.task_config)
            torch.distributed.barrier()
      
        b_text, s_text, h_text = sequence_output.size()
        if visual_output is not None:
            b_modal, s_video, h_video = visual_output.size()
        if audio_output is not None:
            b_modal, s_audio, h_audio = audio_output.size()
       
        
        retrieve_logits_list = []
        step_size = 5 

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]
        

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        

        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]#[step_size, max_text, text_dim]
            attention_mask_row = attention_mask_splits[i]#[step_size, max_text]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_modal, 1, 1)#[step_size, batchsize, max_text, text_dim]
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)#[step_size*batchsize, max_text, text_dim]
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_modal, 1)#[step_size, batchsize, max_text]
            attention_mask_l = attention_mask_l.view(-1, s_text)#[step_size*batchsize, max_text]
           
            input_dict={'sequence_output':sequence_output_l,
                'attention_mask':attention_mask_l}

            step_truth = sequence_output_row.size(0)#step_size
            if visual_output is not None:
                
                visual_output_m = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1) #[step_size, batchsize, max_frames, video_dim]
                visual_output_m = visual_output_m.view(-1, s_video, h_video) #[batchsize*step_size, max_frames, visual_dim]
                video_mask_m = video_mask.unsqueeze(0).repeat(step_truth, 1, 1) #
                video_mask_m = video_mask_m.view(-1, s_video)#[batchsize*step_size, max_frames]
                input_dict.update({'visual_output':visual_output_m, \
                'video_mask':video_mask_m})
            
            if audio_output is not None:
                audio_output_r = audio_output.unsqueeze(0).repeat(step_truth, 1, 1, 1) #[step_size, batchsize, max_frames, video_dim]
                audio_output_r = audio_output_r.view(-1, s_audio, h_audio) #[batchsize*step_size, max_frames, visual_dim]
                audio_mask_r = audio_mask.unsqueeze(0).repeat(step_truth, 1, 1) #
                audio_mask_r = audio_mask_r.view(-1, s_audio)

                input_dict.update({'audio_output':audio_output_r,'audio_mask':audio_mask_r})

            
            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(**input_dict)
            #pooled_output:[step_size* batch_size, dim]
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_modal)#[step_size* batch_size]

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)

        # if self.training:
        #     retrieve_logits= allgather(retrieve_logits, self.task_config)
        #     torch.distributed.barrier()

        #assert torch.isnan(retrieve_logits) is False, logger.info('retrieve_logits gets nan:{}'.format(retrieve_logits))
        return retrieve_logits

    def _tight_seq(self, sequence_output, visual_output=None, audio_output=None,  video_mask=None, audio_mask=None):
        # wating to be filled

        if self.training:
            #print('device:{}, batchsize:{}, audio_output[0]:{}'.format(audio_output.device, audio_output.shape[0],  audio_output[:,0,0]))
            sequence_output = allgather(sequence_output, self.task_config)#[batchsize,1, 512]
            if visual_output is not None: visual_output = allgather(visual_output, self.task_config)
            if video_mask is not None: video_mask = allgather(video_mask, self.task_config)
            if audio_output is not None: audio_output = allgather(audio_output, self.task_config)
            if audio_mask is not None: audio_mask = allgather(audio_mask, self.task_config)
            torch.distributed.barrier()

        attention_mask = torch.ones(sequence_output.size(0),sequence_output.size(1)).to(sequence_output.device)
        b_text, s_text, h_text = sequence_output.size()
        if visual_output is not None: b_modal, s_video, h_video = visual_output.size()
        if audio_output is not None: b_modal, s_audio, h_audio = audio_output.size()
        
        retrieve_logits_list = []
        step_size = 5 

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]
        

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        

        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]#[step_size, max_text, text_dim]
            attention_mask_row = attention_mask_splits[i]#[step_size, max_text]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_modal, 1, 1)#[step_size, batchsize, max_text, text_dim]
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)#[step_size*batchsize, max_text, text_dim]
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_modal, 1)#[step_size, batchsize, max_text]
            attention_mask_l = attention_mask_l.view(-1, s_text)#[step_size*batchsize, max_text]
         
            input_dict={'sequence_output':sequence_output_l,'attention_mask':attention_mask_l }

            step_truth = sequence_output_row.size(0)#step_size
            if visual_output is not None:
                visual_output_m = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1) #[step_size, batchsize, max_frames, video_dim]
                visual_output_m = visual_output_m.view(-1, s_video, h_video) #[batchsize*step_size, max_frames, visual_dim]
                video_mask_m = video_mask.unsqueeze(0).repeat(step_truth, 1, 1) #
                video_mask_m = video_mask_m.view(-1, s_video)#[batchsize*step_size, max_frames]
                input_dict.update({'visual_output':visual_output_m,'video_mask':video_mask_m})
                
            if audio_output is not None:
                audio_output_r = audio_output.unsqueeze(0).repeat(step_truth, 1, 1, 1) #[step_size, batchsize, max_frames, video_dim]
                audio_output_r = audio_output_r.view(-1, s_audio, h_audio) #[batchsize*step_size, max_frames, visual_dim]
                audio_mask_r = audio_mask.unsqueeze(0).repeat(step_truth, 1, 1) #
                audio_mask_r = audio_mask_r.view(-1, s_audio)
                input_dict.update({'audio_output':audio_output_r,'audio_mask':audio_mask_r})

            
            
            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(**input_dict)
            #pooled_output:[step_size* batch_size, dim]
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_modal)#[step_size* batch_size]

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)

        #assert torch.isnan(retrieve_logits) is False, logger.info('retrieve_logits gets nan:{}'.format(retrieve_logits))
        return retrieve_logits
     
    def get_similarity_logits_with_three_modalities(self, sequence_output, visual_output, audio_output, attention_mask, video_mask, audio_mask=None, shaped = False, _pretrain_joint=False, concat_mask=None, input_ids=None, hard_negative=False): 
        
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        
        if self.task_config.retrieval_finetune=='tight_seq':
            if sequence_output.dim() == 3 and  sequence_output.shape[1]>1:
                sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]
                sequence_output = sequence_output.unsqueeze(1)
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
            elif sequence_output.dim() == 2:
                sequence_output = sequence_output.unsqueeze(1)
            t_va_logits = self._tight_seq(sequence_output, visual_output, audio_output,  video_mask, audio_mask)
            t_va_logits *= coffient
            retrieval_logits = {'t_va':t_va_logits}
        elif self.task_config.retrieval_finetune=='cross_align':
            t_va_logits = self._cross_similarity(sequence_output=sequence_output,attention_mask=attention_mask, \
            visual_output=visual_output, audio_output=audio_output,  video_mask=video_mask, audio_mask=audio_mask, concat_mask=concat_mask, input_ids=input_ids)
            t_va_logits *= coffient
            retrieval_logits = {'t_va':t_va_logits}
            

            #train joint after cross
        else:
            retrieval_logits = {}
             
            if self.task_config.retrieval_finetune=='loose_seq':
                concat_features = torch.cat((visual_output, audio_output), dim=1)  # concatnate tokens and frames
                concat_mask = torch.cat((video_mask, audio_mask), dim=1)#[batchsize,896]
                video_type_ = torch.ones_like(video_mask)
                audio_type_ = torch.ones_like(audio_mask) * 2
                audio_type_ = audio_type_.type_as(video_type_)
                concat_type = torch.cat((video_type_, audio_type_), dim=1)
                cross_output, pooled_output, concat_mask = self._get_cross_output(visual_output=visual_output, audio_output=audio_output, video_mask=video_mask, audio_mask=audio_mask)
                visual_output, audio_output = torch.split(cross_output, [video_mask.size(-1), audio_mask.size(-1)], dim=1)
                visual_output = visual_output.contiguous()
                audio_output = audio_output.contiguous()

            if self.training and hard_negative == False:
                #print('device:{}, batchsize:{}, audio_output[0]:{}'.format(audio_output.device, audio_output.shape[0],  audio_output[:,0,0]))
                visual_output = allgather(visual_output, self.task_config)
                video_mask = allgather(video_mask, self.task_config)
                sequence_output = allgather(sequence_output, self.task_config)
                attention_mask = allgather(attention_mask, self.task_config)
                audio_output = allgather(audio_output, self.task_config)
                audio_mask = allgather(audio_mask, self.task_config)
                if input_ids is not None:
                    input_ids = allgather(input_ids, self.task_config)
                torch.distributed.barrier()
    
            if sequence_output.dim() == 3 and  sequence_output.shape[1]>1:
                sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
            else:
                if sequence_output.dim() == 3 and  sequence_output.shape[1]==1:
                    sequence_output = sequence_output.squeeze(1)
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        
            visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True)+1e-10)
            visual_output = self._mean_pooling_for_single_modal(visual_output, video_mask, 'video')
            visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True) + 1e-10)

            audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True)+1e-10)   
            audio_output = self._mean_pooling_for_single_modal(audio_output, audio_mask, 'audio')
            audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True) + 1e-10)
            
            sim_matrix_t_v = torch.matmul(sequence_output, visual_output.t()) * coffient
            sim_matrix_t_a = torch.matmul(sequence_output, audio_output.t()) * coffient
            sim_matrix_v_a = torch.matmul(visual_output, audio_output.t()) * coffient
            if self.task_config.do_pretrain and self.task_config.with_self_supervised and self._stage_one:
                sim_matrix_t_t = torch.matmul(sequence_output, sequence_output.t()) * coffient
                sim_matrix_v_v = torch.matmul(visual_output, visual_output.t()) * coffient
                retrieval_logits.update({
                't_t':sim_matrix_t_t,
                'v_v':sim_matrix_v_v
            }) 

            if self.task_config.retrieval_finetune == 'feat2feat':
                query_weights = self.compute_weights_from_feat(visual_output, audio_output)
                weighted_feat = (query_weights[:,0:1] * visual_output +  query_weights[:,1:2] * audio_output)                    
                query_weights = query_weights.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sim_matrix_t_av = coffient * torch.matmul(sequence_output, weighted_feat.t())

            elif self.task_config.retrieval_finetune == 'query2feat':    
                
                query_weights= self.compute_weights_from_emb(sequence_output,visual_output.shape[0]) #[q,v,2]
                visual_output_expand = visual_output.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                audio_output_expand = audio_output.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sequence_output_expand = sequence_output.unsqueeze(1).repeat(1,visual_output.shape[0],1,)
                weighted_feat = (query_weights[:,:,0:1]  * visual_output_expand +  query_weights[:, :,1:2] * audio_output_expand)
                
                sim_matrix_t_av = coffient * torch.sum(sequence_output_expand *  weighted_feat,dim=-1)
            elif self.task_config.retrieval_finetune == 'feat2sim':
              
                query_weights = self.compute_weights_from_feat(visual_output, audio_output) #[v_batchsize, 2]
                query_weights = query_weights.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sim_matrix_t_av =query_weights[:,:,0] * sim_matrix_t_v + query_weights[:,:,1] * sim_matrix_t_a#[v_b,1]*[s_b,v_b]

            elif self.task_config.retrieval_finetune == 'query2sim':    
                
                query_weights = self.compute_weights_from_emb(sequence_output,visual_output.shape[0])
                sim_matrix_t_av = query_weights[:,:,0]  * sim_matrix_t_v +  query_weights[:, :,1] * sim_matrix_t_a
            
            
            elif self.task_config.retrieval_finetune == 'sim_plus':
                query_weights = torch.ones((sequence_output.shape[0], visual_output.shape[0], 2)) * 0.5
                sim_matrix_t_av = (sim_matrix_t_a + sim_matrix_t_v) /2
            else:
                
                a_v_output = (visual_output + audio_output)/2
                query_weights = torch.ones((sequence_output.shape[0], visual_output.shape[0], 2)) * 0.5
                sim_matrix_t_av = torch.matmul(sequence_output, a_v_output.t()) * coffient
            
                             
            retrieval_logits.update({
                't_v':sim_matrix_t_v,
                't_a':sim_matrix_t_a,
                'a_v':sim_matrix_v_a,
                't_va':sim_matrix_t_av,
                'query_weights':query_weights
            }) 
        
        return retrieval_logits

    def get_similarity_logits(self, modal1_output, modal2_output, modal1_mask, modal2_mask, modal1='text', modal2='video', shaped=False, _pretrain_joint=False,input_ids=None, hard_negative=False):
        '''
        MIL-NCE loss of text sequence and video sequence.
        sequence_output:[batchsize, max_text_len, text_dim=768]
        visual_output:[batchsize, max_frame_len, visual_dim=768]
        attention_mask:[batchsize, max_text_len]
        video_mask:[batchsize, max_frame_len]
        '''

        if shaped is False:
            modal1_mask = modal1_mask.view(-1, modal1_mask.shape[-1])
            modal2_mask = modal2_mask.view(-1, modal2_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        
        if self.task_config.retrieval_finetune=='tight_seq':
            if modal1_output.dim() == 2:
                modal1_output = modal1_output.unsqueeze(1)
            elif modal1_output.dim() == 3 and  modal1_output.shape[1]>1:
                modal1_output = modal1_output[torch.arange(modal1_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]
                modal1_output = modal1_output.unsqueeze(1)
                modal1_output = modal1_output / modal1_output.norm(dim=-1, keepdim=True)
            
            input_dict={'sequence_output':modal1_output}

            if modal2 =='video':
                input_dict.update({'visual_output':modal2_output, 'video_mask':modal2_mask})
            elif modal2 == 'audio':
                input_dict.update({'audio_output':modal2_output, 'audio_mask':modal2_mask})
            sim_matrix = self._tight_seq(**input_dict) * coffient
            
        elif self.task_config.retrieval_finetune=='cross_align':
            input_dict={'sequence_output':modal1_output, 'attention_mask':modal1_mask}
            if modal2 =='video':
                input_dict.update({'visual_output':modal2_output, 'video_mask':modal2_mask})
            elif modal2 == 'audio':
                input_dict.update({'audio_output': modal2_output, 'audio_mask':modal2_mask})

            sim_matrix = self._cross_similarity(**input_dict)
            sim_matrix *= coffient
            
            #train joint after cross
        
        else:
            #[batchsize, text_dim]  [batchsize, visual_dim]
            if self.task_config.retrieval_finetune=='loose_seq':
                
                if modal2 == 'video':  
                    modal2_type = torch.ones_like(modal2_mask)
                    
                if modal2 == 'audio': 
                    modal2_type = torch.ones_like(modal2_mask) * 2
                modal2_type = modal2_type.to(torch.int64)
                modal2_output, pooled_output = self.cross(modal2_output, modal2_type, modal2_mask, output_all_encoded_layers=False)
            
            
            if self.training and hard_negative == False:
                if input_ids is not None:
                    input_ids = allgather(input_ids, self.task_config)
                modal1_output = allgather(modal1_output, self.task_config)
                modal1_mask = allgather(modal1_mask, self.task_config)
                modal2_output = allgather(modal2_output, self.task_config)
                modal2_mask = allgather(modal2_mask, self.task_config)
                torch.distributed.barrier()

            if modal1_output.dim() == 3 and  modal1_output.shape[1]>1:
                modal1_output = modal1_output[torch.arange(modal1_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]] 
                modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)
            else:
                if modal1_output.dim() == 3 and  modal1_output.shape[1]==1:
                    modal1_output = modal1_output.squeeze(1)
                modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)

            modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
            modal2_output = self._mean_pooling_for_single_modal(modal2_output, modal2_mask, modal2)
            modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
            sim_matrix=torch.matmul(modal1_output, modal2_output.t())*coffient
            
        return sim_matrix


from module_adapter.model import MultilingualImageTextModel

class AudioClip_adapter(nn.Module):
    def __init__(self,  task_config,  device='cpu'):
        super(AudioClip_adapter, self).__init__( )
        self.task_config = task_config
        self.ignore_video_index = -1
        self.ignore_audio_index = -1
        self.ignore_index = -2
        self.end_token = {'en':END_TOKEN,'zh':102, 'de':102,'fr':102,'cs':102,'ru':102,'vi':102,'sw':102,'es':102}

        if '9lang' in task_config.adapter_ckpt_path:
            langs = ['de','fr','cs','zh','ru','vi','sw','es']
        else:
            langs = ['de','fr','cs','zh','ja']
        self.clip = MultilingualImageTextModel(task_config.adapter_ckpt_path, langs, device).float()
        embed_dim=512
        image_resolution = 224
        vision_layers = 12
        vision_width = 768
        vision_patch_size = 32
    
        if self.task_config.fp16 == False:
            convert_weights(self.clip)
        # <=== End of CLIP Encoders
        #word_embeddings_weight = self.clip.token_embedding.weight
        #position_embeddings_weight = self.clip.positional_embedding

        # Audio Encoder ===>
     
        self.audio = CLIP4Audio(embed_dim=embed_dim,
        image_resolution = image_resolution, vision_layers = vision_layers, \
            vision_width = vision_width, vision_patch_size = vision_patch_size, \
                with_bg_token=self.task_config.with_bg_token, with_control_token=self.task_config.with_control_token, \
                is_pretrain = self.task_config.do_pretrain).float()
        if self.task_config.fp16 == False:
            convert_weights(self.audio)

        # <=== End of Audio Encoder
        '''
        mil-NCE loss for joint matching(not cross matching)
        '''
        self.nceloss = CrossEn() 

        '''init sub layers of self.init_weights and tied to self'''

        # self.apply(self.init_weights)

    def forward(self, input_txt=None, attention_mask=None, video=None, video_mask=None,
                audio=None, audio_mask=None, input_ids=None, bg_token_gt=None, lan='en'):
    
        self.lan=lan
        sequence_output = self.get_sequence_output(input_txt, attention_mask, lan)
                                                                                                      
        # attention_mask: text token mask
        if video is not None and video_mask is not None:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
            visual_output = self.get_visual_output(video, video_mask,  shaped=True, video_frame=video_frame)
            
        else:
            visual_output = None
            video_mask = None
      
        if audio is not None and audio_mask is not None:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            audio = torch.as_tensor(audio).float()
            b, pair, bs  = audio.shape[:3]
            audio = audio.view(b * pair * bs , *audio.shape[3:]) 
            audio_output, bg_tokens = self.get_audio_output(audio, audio_mask, token_type=bg_token_gt, shaped=True)
        else:
            audio_output = None
            audio_mask = None
        
   
        
        '''
        sequence_output:[batchsize, max_word_len, 512]
        visual_output:[batchsize, max_frame_len, 512]
        audio_output:[batchsize, max_audio_len, 512]
        '''
        if self.training:
            loss_recoder = Loss_recoder()
            loss = 0.
            
            
                #sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]    
             
            if visual_output is None and video_mask is None:  
                sim_matrix_t_a = self.get_similarity_logits(sequence_output, audio_output, attention_mask, audio_mask,'text','audio',input_ids=input_ids, shaped=True)     
            
                sim_loss_t_a = self.nceloss(sim_matrix_t_a)
            
                loss_recoder.update('ta_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_a)
                loss+=sim_loss_t_a
            elif audio_output is None and audio_mask is None:
                sim_matrix_t_v = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,'text','video', input_ids=input_ids,shaped=True)
                        
                sim_loss_t_v = self.nceloss(sim_matrix_t_v) 
                loss_recoder.update('tv_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_v)
                loss+=sim_loss_t_v

            elif self.task_config.retrieval_finetune=='tight_seq' or 'cross_align':
                sim_matrix = self.get_similarity_logits_with_three_modalities(sequence_output, visual_output, audio_output, \
                attention_mask, video_mask, audio_mask,input_ids=input_ids, shaped=True)['t_va']
                sim_loss_t_av_nce= self.nceloss(sim_matrix)
                        
                loss_recoder.update('tav_{}'.format(self.task_config.retrieval_finetune),sim_loss_t_av_nce)
                loss += sim_loss_t_av_nce

            else:
                sim_dict= self.get_similarity_logits_with_three_modalities(sequence_output, visual_output, audio_output, \
                    attention_mask, video_mask, audio_mask,input_ids=input_ids, shaped=True)
                        
                      
                if  'nce' in self.task_config.loss_func:
                    sim_loss_t_v_nce = self.nceloss(sim_dict['t_v']) 
                    sim_loss_t_a_nce = self.nceloss(sim_dict['t_a']) 
                    sim_loss_v_a_nce = (self.nceloss(sim_dict['a_v']) + self.nceloss(sim_dict['a_v'].T)) / 2 
                    sim_loss_t_av_nce = self.nceloss(sim_dict['t_va'])
                            
                    loss_recoder.update('va_nce', sim_loss_v_a_nce)
                    loss_recoder.update('ta_nce', sim_loss_t_a_nce)
                    loss_recoder.update('tv_nce', sim_loss_t_v_nce)
                    loss_recoder.update('tav_nce', sim_loss_t_av_nce)
                    if self.task_config.loss_func == 'ta_nce':
                        loss += sim_loss_t_a_nce 
                    elif self.task_config.loss_func == 'tv_nce':
                        loss += sim_loss_t_v_nce 
                    elif self.task_config.loss_func == 'av_nce':
                        loss += sim_loss_v_a_nce
                    elif self.task_config.loss_func == 'tav_nce':
                        loss += sim_loss_t_av_nce
                    else:
                        raise NotImplementedError
 

            return loss, loss_recoder
        else:
            return None

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)

        visual_hidden = self.clip.encode_image(video).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        
        return visual_hidden

    def get_sequence_output(self, input_txt, attention_mask, lang):
        
        sequence_hidden= self.clip.encode_text(input_txt, attention_mask, lang=lang).float()
        return sequence_hidden
    
    def get_audio_output(self, audio, audio_mask, token_type=None,shaped=False, with_hidden=False):
        if shaped is False:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            
            b, pair, ts = audio.shape[:3]
            audio = audio.view(b * pair*ts , *audio.shape[3:])

        if token_type is not None:
            token_type = token_type.expand(audio_mask.shape).reshape(-1).to(torch.int64)    

        bs_pair = audio_mask.size(0)
        audio_hidden, bg_hidden  = self.audio(audio, token_type = token_type)
        audio_hidden = audio_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        if bg_hidden is not None:
            bg_hidden = bg_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        
        return audio_hidden, bg_hidden
    

    def _mean_pooling_for_single_modal(self, modal_output, modal_mask, modal_type):
        assert modal_type in ['video', 'text', 'audio']
        if modal_type == 'text' :# add an [cls] token
            modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_text_len, 1]
            modal_mask_un[:, 0, :] = 0. #[cls] token
            modal_output = modal_output * modal_mask_un
            modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
            modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
            modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
        #[batchsize, text_dim]
        elif modal_type == 'video' or modal_type == 'audio':
            if modal_output.shape[1] == modal_mask.shape[1]:
                modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_frame_len, 1]
                modal_output = modal_output * modal_mask_un
                modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
                modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
                modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
            else:
                modal_out = modal_output.mean(dim=1)
        #[batchsize, frame_dim]
        
        return modal_out

     
    def get_similarity_logits_with_three_modalities(self, sequence_output, visual_output, audio_output, attention_mask, video_mask, audio_mask=None, shaped = False, _pretrain_joint=False, concat_mask=None, input_ids=None, hard_negative=False): 
        
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        
        if self.task_config.retrieval_finetune=='tight_seq':
            if sequence_output.dim() == 3 and  sequence_output.shape[1]>1:
                sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==self.end_token[self.lan]).nonzero(as_tuple=True)[1]]
                sequence_output = sequence_output.unsqueeze(1)
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
            elif sequence_output.dim() == 2:
                sequence_output = sequence_output.unsqueeze(1)
            t_va_logits = self._tight_seq(sequence_output, visual_output, audio_output,  video_mask, audio_mask)
            t_va_logits *= coffient
            retrieval_logits = {'t_va':t_va_logits}
        elif self.task_config.retrieval_finetune=='cross_align':
            t_va_logits = self._cross_similarity(sequence_output=sequence_output,attention_mask=attention_mask, \
            visual_output=visual_output, audio_output=audio_output,  video_mask=video_mask, audio_mask=audio_mask, concat_mask=concat_mask, input_ids=input_ids)
            t_va_logits *= coffient
            retrieval_logits = {'t_va':t_va_logits}
            

            #train joint after cross
        else:
            retrieval_logits = {}
             
            if self.task_config.retrieval_finetune=='loose_seq':
                concat_features = torch.cat((visual_output, audio_output), dim=1)  # concatnate tokens and frames
                concat_mask = torch.cat((video_mask, audio_mask), dim=1)#[batchsize,896]
                video_type_ = torch.ones_like(video_mask)
                audio_type_ = torch.ones_like(audio_mask) * 2
                audio_type_ = audio_type_.type_as(video_type_)
                concat_type = torch.cat((video_type_, audio_type_), dim=1)
                cross_output, pooled_output, concat_mask = self._get_cross_output(visual_output=visual_output, audio_output=audio_output, video_mask=video_mask, audio_mask=audio_mask)
                visual_output, audio_output = torch.split(cross_output, [video_mask.size(-1), audio_mask.size(-1)], dim=1)
                visual_output = visual_output.contiguous()
                audio_output = audio_output.contiguous()

            if self.training and hard_negative == False:
                #print('device:{}, batchsize:{}, audio_output[0]:{}'.format(audio_output.device, audio_output.shape[0],  audio_output[:,0,0]))
                visual_output = allgather(visual_output, self.task_config)
                video_mask = allgather(video_mask, self.task_config)
                sequence_output = allgather(sequence_output, self.task_config)
                attention_mask = allgather(attention_mask, self.task_config)
                audio_output = allgather(audio_output, self.task_config)
                audio_mask = allgather(audio_mask, self.task_config)
                if input_ids is not None:
                    input_ids = allgather(input_ids, self.task_config)
                torch.distributed.barrier()
    
            if sequence_output.dim() == 3 and  sequence_output.shape[1]>1:
            
                sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==self.end_token[self.lan]).nonzero(as_tuple=True)[1]]
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
            else:
                if sequence_output.dim() == 3 and  sequence_output.shape[1]==1:
                    sequence_output = sequence_output.squeeze(1)
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        
            visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True)+1e-10)
            visual_output = self._mean_pooling_for_single_modal(visual_output, video_mask, 'video')
            visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True) + 1e-10)

            audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True)+1e-10)   
            audio_output = self._mean_pooling_for_single_modal(audio_output, audio_mask, 'audio')
            audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True) + 1e-10)
        
            sim_matrix_t_v = torch.matmul(sequence_output, visual_output.t()) * coffient
            sim_matrix_t_a = torch.matmul(sequence_output, audio_output.t()) * coffient
            sim_matrix_v_a = torch.matmul(visual_output, audio_output.t()) * coffient
            if self.task_config.do_pretrain and self.task_config.with_self_supervised and self._stage_one:
                sim_matrix_t_t = torch.matmul(sequence_output, sequence_output.t()) * coffient
                sim_matrix_v_v = torch.matmul(visual_output, visual_output.t()) * coffient
                retrieval_logits.update({
                't_t':sim_matrix_t_t,
                'v_v':sim_matrix_v_v
            }) 

            if self.task_config.retrieval_finetune == 'feat2feat':
                query_weights = self.compute_weights_from_feat(visual_output, audio_output)
                weighted_feat = (query_weights[:,0:1] * visual_output +  query_weights[:,1:2] * audio_output)                    
                query_weights = query_weights.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sim_matrix_t_av = coffient * torch.matmul(sequence_output, weighted_feat.t())

            elif self.task_config.retrieval_finetune == 'query2feat':    
                
                query_weights= self.compute_weights_from_emb(sequence_output,visual_output.shape[0]) #[q,v,2]
                visual_output_expand = visual_output.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                audio_output_expand = audio_output.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sequence_output_expand = sequence_output.unsqueeze(1).repeat(1,visual_output.shape[0],1,)
                weighted_feat = (query_weights[:,:,0:1]  * visual_output_expand +  query_weights[:, :,1:2] * audio_output_expand)
                
                sim_matrix_t_av = coffient * torch.sum(sequence_output_expand *  weighted_feat,dim=-1)
            elif self.task_config.retrieval_finetune == 'feat2sim':
              
                query_weights = self.compute_weights_from_feat(visual_output, audio_output) #[v_batchsize, 2]
                query_weights = query_weights.unsqueeze(0).repeat(sequence_output.shape[0],1,1)
                sim_matrix_t_av =query_weights[:,:,0] * sim_matrix_t_v + query_weights[:,:,1] * sim_matrix_t_a#[v_b,1]*[s_b,v_b]

            elif self.task_config.retrieval_finetune == 'query2sim':    
                
                query_weights = self.compute_weights_from_emb(sequence_output,visual_output.shape[0])
                sim_matrix_t_av = query_weights[:,:,0]  * sim_matrix_t_v +  query_weights[:, :,1] * sim_matrix_t_a
            
            
            elif self.task_config.retrieval_finetune == 'sim_plus':
                query_weights = torch.ones((sequence_output.shape[0], visual_output.shape[0], 2)) * 0.5
                sim_matrix_t_av = (sim_matrix_t_a + sim_matrix_t_v) /2
            else:
                
                a_v_output = (visual_output + audio_output)/2
                query_weights = torch.ones((sequence_output.shape[0], visual_output.shape[0], 2)) * 0.5
                sim_matrix_t_av = torch.matmul(sequence_output, a_v_output.t()) * coffient
            
                             
            retrieval_logits.update({
                't_v':sim_matrix_t_v,
                't_a':sim_matrix_t_a,
                'a_v':sim_matrix_v_a,
                't_va':sim_matrix_t_av,
                'query_weights':query_weights
            }) 
        
        return retrieval_logits

    def get_similarity_logits(self, modal1_output, modal2_output, modal1_mask, modal2_mask, modal1='text', modal2='video', shaped=False, _pretrain_joint=False,input_ids=None, hard_negative=False):
        '''
        MIL-NCE loss of text sequence and video sequence.
        sequence_output:[batchsize, max_text_len, text_dim=768]
        visual_output:[batchsize, max_frame_len, visual_dim=768]
        attention_mask:[batchsize, max_text_len]
        video_mask:[batchsize, max_frame_len]
        '''

        if shaped is False:
            modal1_mask = modal1_mask.view(-1, modal1_mask.shape[-1])
            modal2_mask = modal2_mask.view(-1, modal2_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        
        if self.task_config.retrieval_finetune=='tight_seq':
            if modal1_output.dim() == 2:
                modal1_output = modal1_output.unsqueeze(1)
            elif modal1_output.dim() == 3 and  modal1_output.shape[1]>1:
                modal1_output = modal1_output[torch.arange(modal1_output.shape[0]), (input_ids==self.end_token[self.lan]).nonzero(as_tuple=True)[1]]
                modal1_output = modal1_output.unsqueeze(1)
                modal1_output = modal1_output / modal1_output.norm(dim=-1, keepdim=True)
            
            input_dict={'sequence_output':modal1_output}

            if modal2 =='video':
                input_dict.update({'visual_output':modal2_output, 'video_mask':modal2_mask})
            elif modal2 == 'audio':
                input_dict.update({'audio_output':modal2_output, 'audio_mask':modal2_mask})
            sim_matrix = self._tight_seq(**input_dict) * coffient
            
        elif self.task_config.retrieval_finetune=='cross_align':
            input_dict={'sequence_output':modal1_output, 'attention_mask':modal1_mask}
            if modal2 =='video':
                input_dict.update({'visual_output':modal2_output, 'video_mask':modal2_mask})
            elif modal2 == 'audio':
                input_dict.update({'audio_output': modal2_output, 'audio_mask':modal2_mask})

            sim_matrix = self._cross_similarity(**input_dict)
            sim_matrix *= coffient
            
            #train joint after cross
        
        else:
            #[batchsize, text_dim]  [batchsize, visual_dim]
            if self.task_config.retrieval_finetune=='loose_seq':
                
                if modal2 == 'video':  
                    modal2_type = torch.ones_like(modal2_mask)
                    
                if modal2 == 'audio': 
                    modal2_type = torch.ones_like(modal2_mask) * 2
                modal2_type = modal2_type.to(torch.int64)
                modal2_output, pooled_output = self.cross(modal2_output, modal2_type, modal2_mask, output_all_encoded_layers=False)
            
            
            if self.training and hard_negative == False:
                if input_ids is not None:
                    input_ids = allgather(input_ids, self.task_config)
                modal1_output = allgather(modal1_output, self.task_config)
                modal1_mask = allgather(modal1_mask, self.task_config)
                modal2_output = allgather(modal2_output, self.task_config)
                modal2_mask = allgather(modal2_mask, self.task_config)
                torch.distributed.barrier()

            if modal1_output.dim() == 3 and  modal1_output.shape[1]>1:
                modal1_output = modal1_output[torch.arange(modal1_output.shape[0]), (input_ids==self.end_token[self.lan]).nonzero(as_tuple=True)[1]] 
                modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)
            else:
                if modal1_output.dim() == 3 and  modal1_output.shape[1]==1:
                    modal1_output = modal1_output.squeeze(1)
                modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)

            modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
            modal2_output = self._mean_pooling_for_single_modal(modal2_output, modal2_mask, modal2)
            modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
            sim_matrix=torch.matmul(modal1_output, modal2_output.t())*coffient
            
        return sim_matrix
