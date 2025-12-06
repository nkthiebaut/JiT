INPUT_PATH=/home/jovyan/data/scene_text_translation/datasets/tiny-imagenet
OUTPUT_DIR=/home/jovyan/data/scene_text_translation/datasets/tiny-imagenet/training_runs

# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
python -m pdb -c c ../main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 64 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 10 --warmup_epochs 5 \
--gen_bsz 128 --num_images 5000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${INPUT_PATH} --online_eval \
--wandb --wandb_project JiT-training