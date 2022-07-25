import torch
import logging
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_Retrieval_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_Retrieval_TrainDataLoader


logger = logging.getLogger(__name__)

def dataloader_msrvtt_retrieval_train(args,tokenizer):
    msrvtt_dataset = MSRVTT_Retrieval_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        unfold_sentences=args.expand_msrvtt_sentences,
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,  
        filter_video_id = args.filter_video_id    
        
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_retrieval_test(args, tokenizer):
    msrvtt_testset = MSRVTT_Retrieval_DataLoader(
        json_path=args.val_csv,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
       
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = msrvtt_testset.get_meta()
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset), meta
