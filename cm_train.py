"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    # create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)

from util.model_util import create_model_and_diffusion
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy

from posa.dataset import ProxDataset_txt, HUMANISE
from torch.utils.data import DataLoader

def local_load_data(loader):
    while True:
        yield from loader

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    # model_and_diffusion_kwargs = args_to_dict(
    #     args, model_and_diffusion_defaults().keys()
    # )

    datatype = "proxd"

    # model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(datatype)
    # model.to(dist_util.dev())
    # model.train()
    # if args.use_fp16:
    #     model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    # train_data_loader = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )

    train_data_dir = "data/protext/proxd_train"
    valid_data_dir = "data/protext/proxd_valid"
    max_frame = 256
    jump_step = 8
    fix_ori = True
    num_workers = 0
    if datatype == "proxd":
        print("this is proxd")
        train_dataset = ProxDataset_txt(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                    step_multiplier=1, jump_step=jump_step)
        train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        valid_dataset = ProxDataset_txt(valid_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                    step_multiplier=1, jump_step=jump_step)
        valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    train_data_loader = local_load_data(train_data_loader)

    ## Currently we don't have teacher model so ignore this code =======
    # if len(args.teacher_model_path) > 0:  # path to the teacher score model.
    #     logger.log(f"loading the teacher model from {args.teacher_model_path}")
    #     teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
    #     teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
    #     teacher_model_and_diffusion_kwargs["distillation"] = False
    #     teacher_model, teacher_diffusion = create_model_and_diffusion(
    #         **teacher_model_and_diffusion_kwargs,
    #     )

    #     teacher_model.load_state_dict(
    #         dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
    #     )

    #     teacher_model.to(dist_util.dev())
    #     teacher_model.eval()

    #     for dst, src in zip(model.parameters(), teacher_model.parameters()):
    #         dst.data.copy_(src.data)

    #     if args.use_fp16:
    #         teacher_model.convert_to_fp16()

    # else:
    #     teacher_model = None
    #     teacher_diffusion = None

    # load the target model for distillation, if path specified.
    ## =============================================================
    logger.log("creating the target model")

    target_model, _ = create_model_and_diffusion(datatype)
    print("done creating target_model")


    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    # if args.use_fp16:
    #     target_model.convert_to_fp16()

    logger.log("training...")

    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=None,                    # currently we don't use teacher model and diffusion
        teacher_diffusion=None,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=train_data_loader,
        batch_size=batch_size,
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


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
