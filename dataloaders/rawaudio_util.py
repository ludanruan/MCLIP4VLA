import soundfile as sf
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')

def audio_processor(audio_frames):
    '''
    normalize the raw audio_frames:[len,3,224,224]
    '''
    # mean=(0.48145466, 0.4578275, 0.40821073),
    
    # std=(0.26862954, 0.26130258, 0.27577711)
    mu = np.mean(audio_frames, axis=(2,3))[:, :, None, None]
    sigma = np.std(audio_frames, axis=(2,3))[:, :, None, None] + 1e-10
    audio_frames = (audio_frames - mu)/sigma
    return audio_frames

def get_raw_audio(audio_path,target_rate, start=None, end=None, max_s=60):
    if start is not None and end is not None:
        start_f,  end_f = target_rate*start, target_rate*end
        audio_wav, audio_rate = sf.read(audio_path,start=start_f, stop=end_f)
    else:
        audio_wav, audio_rate = sf.read(audio_path)    
   
    max_audio_len = max_s * audio_rate
    if audio_wav.shape[0] > max_audio_len:
      start = int(audio_wav.shape[0]/2 - max_audio_len/2)
      end = int( start + max_audio_len)
      audio_wav = audio_wav[start:end,:]

    if target_rate < audio_rate:
      #down sampling
      audio_wav = librosa.resample(audio_wav, audio_rate, target_rate)

    if audio_wav.ndim == 1: 
      audio_wav = audio_wav[:, np.newaxis]
      audio_wav = np.repeat(audio_wav,2,axis=1)

    return audio_wav  

def wav2fbank(waveform, audio_rate):
    # get wavform and turn to fbank every audio frame 224*224
    # waveform:[1,]
    
    melspec= librosa.feature.melspectrogram(waveform, audio_rate, n_fft=512, hop_length=128, n_mels=224, fmax=8000)
    logmelspec = librosa.power_to_db(melspec)

    
    return logmelspec.T

def split_frame(audio_frame, overlap=0, single_frame_len=224):
     
    audio_pad = single_frame_len - audio_frame.shape[1] % single_frame_len
           
    if audio_pad > 0:
        zero_pad = np.zeros([3, audio_pad, single_frame_len])
        audio_frame = np.concatenate([audio_frame, zero_pad], axis=1)
    audio_frame = audio_frame.reshape(3, -1, single_frame_len, single_frame_len).transpose(1,0,2,3)
    return audio_frame


# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     audio_path = '/dataset/28d47491/rld/data/msrvtt/audios_16k/video284.wav'
    
#     audio_wav = get_raw_audio(audio_path, 16000)
#     audio_frame_l = wav2fbank(audio_wav[:,0],16000)
#     audio_frame_r = wav2fbank(audio_wav[:,1],16000)
#     audio_frame_m = wav2fbank((audio_wav[:, 0]+audio_wav[:, 1])/2,16000)

#     pdb.set_trace()
#     audio_frame = np.stack([audio_frame_l, audio_frame_m, audio_frame_r], axis=0)
   
    
#     audio_frame_aug = spec_augment(audio_frame)
#     audio_frame_aug = split_frame(audio_frame, overlap=0, single_frame_len=224)
#     audio_frame_aug = audio_processor(audio_frame_aug)

#     audio_frame = split_frame(audio_frame, overlap=0, single_frame_len=224)
#     audio_frame = audio_processor(audio_frame)


    
