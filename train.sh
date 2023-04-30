
MODEL_FLAGS="--in_channels 3 --out_channels 6 --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --class_cond False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

TRAIN_FLAGS="--lr 1e-4 --batch_size 4"

DATA_PATH="./DATASETS/dlss21_ho4_data/train"

python3 scripts/segmentation_train.py --data_dir "$DATA_PATH" $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS

