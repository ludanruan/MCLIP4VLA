import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import math
import argparse
import logging
from typing import Dict, List, Union, Tuple, Any
from pathlib import Path

from modules.optimization import BertAdam

def frame_signal(signal: torch.Tensor,
                 frame_length: int,
                 hop_length: int,
                 window: torch.Tensor = None) -> torch.Tensor:

    if window is None:
        window = torch.ones(frame_length, dtype=signal.dtype, device=signal.device)

    if window.shape[0] != frame_length:
        raise ValueError('Wrong `window` length: expected {}, got {}'.format(window.shape[0], frame_length))

    signal_length = signal.shape[-1]

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_length) / hop_length))

    pad_len = int((num_frames - 1) * hop_length + frame_length)
    if pad_len > signal_length:
        zeros = torch.zeros(pad_len - signal_length, device=signal.device, dtype=signal.dtype)

        while zeros.dim() < signal.dim():
            zeros.unsqueeze_(0)

        pad_signal = torch.cat((zeros.expand(*signal.shape[:-1], -1)[..., :zeros.shape[-1] // 2], signal), dim=-1)
        pad_signal = torch.cat((pad_signal, zeros.expand(*signal.shape[:-1], -1)[..., zeros.shape[-1] // 2:]), dim=-1)
    else:
        pad_signal = signal

    indices = torch.arange(0, frame_length, device=signal.device).repeat(num_frames, 1)
    indices += torch.arange(
        0,
        num_frames * hop_length,
        hop_length,
        device=signal.device
    ).repeat(frame_length, 1).t_()
    indices = indices.long()

    frames = pad_signal[..., indices]
    frames = frames * window

    return frames

def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def get_less_multiple(int_a, int_b):
    if int_a >= int_b:
        greater, minor = int_a, int_b
        
    else:
        greater, minor = int_b, int_a
   
    
    multi=1
    while True:
        if multi*greater % minor == 0:
            return multi * greater
        else:
            multi += 1

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    '''
    ftc:_run_on_single_gpu: function
    model:UniVL or UnivL_audio
    inputs: len(input)=6(include aucio) or 4(exceopt audio)
    device_id: gpu ids (0,1,2,3)
    '''
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

def prep_optimizer_clip(args, model, num_train_optimization_steps, device, n_gpu, local_rank):

    if hasattr(model, 'module'):
        model = model.module
    if hasattr(args, 'langs') and args.langs == 'en':
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer=[]
        for key, val in list(model.named_parameters()):
            if 'adapter' in key:
                param_optimizer.append([key, val])
    
    if args.pretrained_clip_name.startswith('ViT'):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    elif  args.pretrained_clip_name.startswith('RN'):
        no_decay =[ 'logits_scale', 'audio.fbsp',  'bias', 'bn']
    else:
        pass

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    
    decay_text_param_tp = [(n, p) for n, p in decay_param_tp if  n.startswith("clip.text_encoder")]
    decay_audio_param_tp = [(n, p) for n, p in decay_param_tp if n.startswith("audio.")]
    decay_visual_param_tp = [(n, p) for n, p in decay_param_tp if (n.startswith("clip.text_encoder") or n.startswith("audio."))==False ]

    no_decay_text_param_tp = [(n, p) for n, p in no_decay_param_tp if n.startswith("clip.text_encoder")]
    no_decay_audio_param_tp = [(n, p) for n, p in no_decay_param_tp if n.startswith("audio.")]
    no_decay_visual_param_tp = [(n, p) for n, p in no_decay_param_tp if (n.startswith("clip.text_encoder") or n.startswith("audio."))==False ]
    

    weight_decay=0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_text_param_tp], 'weight_decay': weight_decay, 'lr': args.lr_t},
        {'params': [p for n, p in decay_audio_param_tp], 'weight_decay': weight_decay, 'lr':args.lr_a },
        {'params': [p for n, p in decay_visual_param_tp], 'weight_decay': weight_decay, 'lr':args.lr_v },
        {'params': [p for n, p in no_decay_text_param_tp], 'weight_decay': 0.0, 'lr':args.lr_t },
        {'params': [p for n, p in no_decay_audio_param_tp], 'weight_decay': 0.0, 'lr':args.lr_a },
        {'params': [p for n, p in no_decay_visual_param_tp], 'weight_decay': 0.0, 'lr':args.lr_v },
        
    ]
    
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr_v, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

class mylog(object):
    def __init__(self, filename=None):
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
        self.logger_path = filename
        if filename is not None:
            handler = logging.FileHandler(self.logger_path, 'a')
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logging.getLogger().addHandler(handler)

    def getlog(self):
        return self.logger

    def reset_handler(self):
        logging.shutdown()
        handler = logging.FileHandler(self.logger_path)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)

    def info(self, *args):
        
        try:
            self.logger.info(*args)
        except:
            print("reset logger handle")
            self.reset_handler()
            self.logger.info(*args)
        return

    def warning(self, *args):
        
        try:
            self.logger.warning(*args)
        except:
            print("reset logger handle")
            self.reset_handler()
            self.logger.warning(*args)
        return

def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
   
    parser.add_argument('--multi_sentence', type=int, default=1, help='multi text to one video retrieval')

    parser.add_argument('--train_csv', type=str, default='', help='video ids of training set')
    parser.add_argument('--val_csv', type=str, default='', help='video ids of validation set')
    

    parser.add_argument('--data_path', type=str, default='data/msrvtt/annotations/multilingual_train/ref_captions_all.json',
                        help='caption and transcription json file path')
    parser.add_argument('--frame_path', type=str, default='./data/msrvtt/raw_frames', help='frame path')
    parser.add_argument('--audio_path', type=str, default='./data/msrvtt/audios_16k', help='audio path')
    parser.add_argument('--raw_video_path', type=str, default=None, help='video path for visualization')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr_a', type=float, default=0.00001, help='learning rate of audio backbone')
    parser.add_argument('--lr_v', type=float, default=0.00001, help='learning rate of video backbone')
    parser.add_argument('--lr_t', type=float, default=0.00001, help='learning rate of text backbone')

    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--audio_rate', type=int, default=16000, help='audio fps')
    parser.add_argument('--audio_channel', type=int, default=2, help='audio channels(1 or 2)')
    parser.add_argument('--audio_tokenlen', type=float, default=1, help='audio feature token length')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--max_audio_length', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--with_control_token', type=float, default=0.5, help='')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")


    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument('--loss_func', type=str, default='nce')
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")
    parser.add_argument('--type_vocab_size', type=int, default=3, help="")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_model", default="pytorch_model.bin.checkpoint", type=str, required=False,
                        help="Save the last model as a checkpoint.")
    
    
    # parameters for MultilingualClip ====>
    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--freeze", default=None, type=str, help="freeze model with key word")
    #<==== parameters for MultilingualClip


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    labels = torch.randn(5,5)
    labels[4,:]= labels[0,:]
    print(build_gt(labels))