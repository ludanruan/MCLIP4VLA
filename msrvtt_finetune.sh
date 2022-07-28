
TRAIN_CSV="data/msrvtt/annotations/MSRVTT_train.7k.csv"
VAL_CSV="data/msrvtt/annotations/multilingual_test/test_en.json"
DATA_PATH="data/msrvtt/annotations/multilingual_train/ref_captions_all.json"
AUDIO_PATH="data/msrvtt/audios_16k"
VIDEO_PATH="data/msrvtt/videos"
OUTPUT_ROOT="ckpts"
FRAME_PATH="data/msrvtt/raw_frames"
INIT_MODEL="weights/MCLIP4VLA.pt"


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25200 \
main_task_video_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=100  \
--pretrained_clip_name ViT-B/32  \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}   --with_control_token 0.5 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--frame_path ${FRAME_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_video_retrieval   \
--lr_a 1e-7 --lr_v 1e-7 --lr_t 1e-5 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1   \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce \
--max_audio_length=6   \



for i in '0' '1' '2' '3' '4'
do
for language in 'en' 'zh' 'cs' 'de' 'es' 'fr' 'ru' 'sw' 'vi'
do
INIT_MODEL="ckpts/ckpt_msrvtt_video_retrieval/pytorch_model.bin.${i}"
VAL_CSV="data/msrvtt/annotations/multilingual_test/test_${language}.json"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25203 \
main_task_video_retrieval.py --do_eval  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=100  \
--pretrained_clip_name ViT-B/32  \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --multilingual_init ${MULTILINGAL_INIT} --with_control_token 0.5 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_video_retrieval  \
--datatype ${DATATYPE} \
--lr_a 1e-7 --lr_v 1e-7 --lr_t 1e-5 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1   \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func ta_nce \
--max_audio_length=6   

done
done
