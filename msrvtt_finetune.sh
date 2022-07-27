DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.7k.csv"
VAL_CSV="data/msrvtt/annotations/multilingual_test/test_en.json"
DATA_PATH="data/msrvtt/annotations/multilingual_train/ref_captions_all.json"
AUDIO_PATH="data/msrvtt/audios_16k"
VIDEO_PATH="data/msrvtt/videos"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="data/msrvtt/raw_frames"

INIT_MODEL="weights/MCLIP4VLA.pt"


CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25200 \
main_task_video_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=64 --n_display=100  \
--pretrained_clip_name ViT-B/32  \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}   --with_control_token 0.5 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_video_retrieval/multilingual_zeroshot   \
--datatype ${DATATYPE} \
--lr_a 1e-7 --lr_v 1e-7 --lr_t 1e-5 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1   \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce \
--max_audio_length=6   \