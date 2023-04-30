"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append(".")
import numpy as np
import torch as th
from tqdm import tqdm
from skimage import io
import datetime



from guided_diffusion import dist_util, logger
# from guided_diffusion.bratsloader import BRATSDataset
from data_loading import MuscleDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion import visualization_utils as viz_utils


seed=10
# th.manual_seed(seed)
# th.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)

def get_dir_name() :
    curr = datetime.datetime.now()
    dm = {1:'jan', 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun", 7:"jul", 8:"aug",
          9:"sep", 10:"nov", 11:"oct", 12:"dec"}
    return f'sampling_{curr.day}_{dm[curr.month]}_{curr.hour}h{curr.minute}m{curr.second}s'


def create_numpy_niced_dir(dir_name, niced=True, npy=True) :
    np_path, nice_path = None, None
    if npy :
        np_path = os.path.join(dir_name, f"numpy_arrays")
        os.makedirs(np_path)
    if niced :
        nice_path = os.path.join(dir_name, f"niced")
        os.makedirs(nice_path)
    return np_path, nice_path


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    seed = args.random_state
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    ds = MuscleDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)



    data = iter(datal)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dir_name = get_dir_name()
    os.makedirs(dir_name)

    if args.tb_display :
        tb_writer = diffusion.set_tb_writer(os.path.join('.', *['runs', dir_name.replace(os.sep, '_')]))

    np_path, nice_path = create_numpy_niced_dir(dir_name, niced=args.save_niced, npy=args.save_numpy)

    for input, fid in tqdm(data) :
        fid = fid[0]
        c = th.randn((input.shape[0], args.out_channels, input.shape[2], input.shape[3]))
        normalized = viz_utils.visualize(input).squeeze().cpu()
        for chan in range(args.in_channels) :
            input_name = os.path.join(dir_name, f"{fid}_0_{chan}.jpg")
            io.imsave(input_name, np.array(normalized[chan, ...], dtype=np.uint8), check_contrast=False)

        merged = th.max(c, dim=1)[1] # take argmax
        noise_name = os.path.join(dir_name, f"{fid}_noise_0.jpg")
        io.imsave(noise_name, np.array(merged.squeeze().cpu(), dtype=np.uint8), check_contrast=False)

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        img = th.cat((input, c), dim=1) #add a noise channel$

        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of num_ensemble masks.
            model_kwargs = {}
            start.record()
            # build the sample function :
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            # sample : (with 1000 step usually)
            sample, _, _ = sample_fn(
                model=model,
                shape = (args.batch_size, args.in_channels + args.out_channels, args.image_size, args.image_size),
                img=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                fid=fid,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            # Saving :
            sample_name = f"{fid}_sample_{i}"
            sample_path = os.path.join(dir_name, f"{sample_name}.jpg")

            # save merge (H,W) image where pixels intensity are classes
            merged = th.max(sample, dim=1)[1]
            io.imsave(sample_path, np.array(merged.squeeze().cpu(), dtype=np.uint8), check_contrast=False)

            # Save multi channels, raw output
            if np_path is not None :
                with open(os.path.join(np_path, f"{sample_name}.npy"), "wb") as f :
                    np.save(f, sample.cpu()) # save sample not merged yet
            # save a nice colored version of the mask
            if nice_path is not None :
                niced = viz_utils.make_displayable(sample.squeeze().cpu())
                io.imsave(os.path.join(nice_path, f"{sample_name}.png"), niced, check_contrast=False)


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,      #number of samples in the ensemble
        in_channels=5,
        out_channels=5,
        class_cond = False,
        random_state=1507,
        save_niced=True,
        save_numpy=True,
        tb_display=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()