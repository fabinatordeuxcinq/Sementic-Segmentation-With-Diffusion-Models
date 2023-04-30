"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from data_loading import MuscleDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import time
import torch as th
from guided_diffusion.train_util import TrainLoop

def main():

    args = create_argparser().parse_args()

    seed = args.random_state
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    # create your own Dataloader
    ds = MuscleDataset(args.data_dir, test_flag=False)

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    # track whole training time
    start_time = time.time()
    data = iter(datal)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    end = time.time() - start_time
    logger.log(f"Whole time {end} (s)")


def create_argparser():

    defaults = dict(
        data_dir="./data/training",
        schedule_sampler="uniform",
        random_state=1507, # random state
        lr=1e-4, # learing rate
        weight_decay=0.0,
        lr_anneal_steps=0,
        training_steps=60_000,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10_000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        in_channels=1, # number of modalities as input (3 for RGB image, 1 for grayscale)
        out_channels=1,# number of classes of the semantic segmentation problem
        class_cond = False,
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
