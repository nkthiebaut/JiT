INPUT_PATH=/home/jovyan/data/scene_text_translation/datasets/tiny-imagenet
OUTPUT_DIR=/home/jovyan/data/scene_text_translation/datasets/tiny-imagenet/training_runs/tiny_run-7

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
../main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 64 --noise_scale 1.0 \
--batch_size 256 --blr 1e-5 \
--epochs 200 --warmup_epochs 5 \
--class_num 200 \
--gen_bsz 256 --num_images 5000  --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--num_sampling_steps 50 \
--output_dir ${OUTPUT_DIR} \
--data_path ${INPUT_PATH} \
--online_eval \
--sampling_method euler \
--wandb \
--wandb_project JiT-training

# --resume ${OUTPUT_DIR} \