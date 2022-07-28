import argparse
import json
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
from joblib import delayed
from joblib import Parallel
from moviepy.editor import VideoFileClip
from dataloaders.rawvideo_util import RawVideoExtractor
import soundfile as sf
import pandas as pd

VIDEO_SUFFIX=['mp4','webm','mkv']
def extract_audios_wrapper(audio_in_dir, audio_in,  output_dir):
    """Wrapper for parallel processing purposes."""
    
    if isinstance(audio_in, np.bytes_):
        audio_in = audio_in.decode()
    
    #output_filename = construct_video_filename(row, trim_format)
    audio_in_path = os.path.join(audio_in_dir, audio_in)
    audio_in_suffix=audio_in.split('.')[-1]
    
    audio_out = os.path.join(output_dir, audio_in.replace(audio_in_suffix,'wav'))
    
    
    if os.path.exists(audio_out):
        status = tuple([audio_out, True, 'Exists'])
        return status
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
   
    try:
       
        audio_handle = VideoFileClip(audio_in_path).audio
        audio_handle.write_audiofile(audio_out, fps=16000,  logger=None)
        #use sf to compress audio
        wav, rate = sf.read(audio_out)
        status = tuple([audio_out, "successful"])
        if wav.mean() == 0:
            print(audio_out, ' wav is blank', )
            os.system('rm '+audio_out)
            status = tuple([audio_out, 'failed' , ' wav is blank'])
        

    except Exception as e:
        print(audio_out, ' ', e.args)
        if os.path.exists(audio_out):
            os.system('rm '+audio_out)
        status = tuple([audio_out, 'failed' , e.args])
 
    return status

def extract_audios(input_dir, output_dir, num_jobs=20):
    '''transfer video to audio from input dir to output_dir with num_jobs threads,
       following the file tree orgnization of input_dir '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("prepare audio list...")    
    audio_list = os.listdir(input_dir)
    
    # Transfer all videos to audios.
    if num_jobs == 1:
        status_lst = []
        for row in tqdm(audio_list):
            status_lst.append(extract_audios_wrapper(input_dir, row, output_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(extract_audios_wrapper)(
           input_dir, row, output_dir) for row in tqdm(audio_list))


def load_video_into_frames_wrapper(video_path, frame_root, feature_framerate = 1, image_resolution=224):
    
    rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
    
    video_name = video_path.split('/')[-1]
    video_id = video_name.split('.')[0]
    
    frame_dir = os.path.join(frame_root,  video_id)
    
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)

    os.makedirs(frame_dir,exist_ok = True)

    try:
        raw_video_slices = rawVideoExtractor.extract_frames(video_path, sample_fp = 1)
           
    except:
        print("read {} failed".format(video_path))
        raw_video_slices = None
        shutil.rmtree(frame_dir)
        return [video_name, False]

   
    if raw_video_slices is not None:
        for idx, frame in enumerate(raw_video_slices):
            frame_path = os.path.join(frame_dir, str(idx)+'.jpg')
            cv2.imwrite(frame_path, frame)
       
    return [video_name, True]

def load_video_into_frames(video_dir, frame_dir, num_jobs=1):
    
    status_lst = []
    
    video_names = os.listdir(video_dir)
    if num_jobs == 1:
        for video_name in tqdm(video_names):
            video_path= os.path.join(video_dir, video_name)
            status_lst.append(load_video_into_frames_wrapper(video_path, frame_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(load_video_into_frames_wrapper)(
            os.path.join(video_dir, video_name), frame_dir) for video_name in tqdm(video_names))       

    print("extracting frames from {} to {} has finished".format(video_dir, frame_dir))       
    return

def build_audio_bash(bash_in, bash_out):
    bash_commands=[]
    with open(bash_in, 'r') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if line.endswith('wav') == False: continue
            subtokens = line.split(' ')
            
            wav_to, wav_from = subtokens[14], subtokens[16]
            wav_from = wav_from.split('/')[-1]
            bash_command=' '.join(["ln", "-s",  wav_from,  wav_to])
            bash_commands.append(bash_command)

    with open(bash_out, 'w') as f_out:
        for bash_command in bash_commands:
            f_out.write(bash_command+"\n")

    print(f"output {bash_out} successfully and plz use it to replace the blank audios")

if __name__ == '__main__':
    description = 'Processors for msrvtt '
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--extract_audios', action='store_true',
                   help='The task will transfer videos to audios that can be read by soundfile')
    p.add_argument('--load_video_into_frames', action='store_true',
                   help='The task will transfer video  into frames')
    p.add_argument('--build_audio_bash', action='store_true', 
                    help='build audio soft links')

    p.add_argument('--video_dir', type=str, default="../data/msrvtt/videos",
                   help=('video dir'))
    
    p.add_argument('--audio_dir', type=str, default="../data/msrvtt/audios_16k",
                   help='Output directory where audios to be saved')
    p.add_argument('--frame_dir', type=str, default="../data/msrvtt/raw_frames",
                   help='Output directory where raw frames to be saved')

    p.add_argument('--bash_in', type=str, default="../data/msrvtt_temp/softlink.sh",
                   help='Output directory where audios to be saved')
    p.add_argument('--bash_out', type=str, default="../data/msrvtt/softlink.sh",
                   help='Output directory where raw frames to be saved')    

    p.add_argument('-n', '--num_jobs', type=int, default=36)
    args= p.parse_args()


    if args.extract_audios:
        assert args.audio_dir is not None  and args.video_dir is not None 
        extract_audios(args.video_dir, args.audio_dir,  args.num_jobs)
    
    if args.load_video_into_frames:
        assert args.video_dir is not None and args.frame_dir is not None and args.num_jobs is not None
        load_video_into_frames(args.video_dir, args.frame_dir, args.num_jobs)
        
    if args.build_audio_bash:
        build_audio_bash(args.bash_in, args.bash_out)
    
    