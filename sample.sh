
MODEL_FLAGS="--in_channels 3 --out_channels 6 --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --class_cond False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --save_niced True --save_numpy True --tb_display True --rescale_learned_sigmas False --rescale_timesteps False "

DATA_PATH="./DATASETS/dlss21_ho4_data/test"

MODEL_PATH="./results/savedmodel000000.pt"

NUM_SAMPLES=5

python3 scripts/segmentation_sample.py  --data_dir "$DATA_PATH" --model_path "$MODEL_PATH"  --num_ensemble $NUM_SAMPLES $MODEL_FLAGS $DIFFUSION_FLAGS
