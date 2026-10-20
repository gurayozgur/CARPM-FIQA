"""CARPM-FIQA training: FR training (ArcFace / CurricularFace) with a quality-regression head
that regresses the cumulative average of the relative point margin (CCS / NNCCS) of every
training sample over all epochs seen so far (Algorithm 1 in the paper).

Launch with torchrun, e.g. (4 GPUs, 128 images per GPU = global batch 512):
    torchrun --nproc_per_node=4 train.py --dataset webface --data_root /data/faces_webface_112x112 --output output/CARPM-FIQA-S
"""
import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn import CrossEntropyLoss
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

import losses
from backbones.iresnet import iresnet50, iresnet100
from dataset import DataLoaderX, ImageFolderDataset, MXFaceDataset

DATASETS = {
    # CARPM-FIQA(S): ResNet50 on CASIA-WebFace (image folders), 34 epochs
    "webface": dict(network="iresnet50", num_classes=10572, num_epoch=34, lr_milestones=[20, 28, 32]),
    # CARPM-FIQA(L): ResNet100 on MS1MV2 (InsightFace train.rec / train.idx), 18 epochs
    "ms1mv2": dict(network="iresnet100", num_classes=85742, num_epoch=18, lr_milestones=[8, 14, 20, 25]),
}


def parse_args():
    parser = argparse.ArgumentParser(description="CARPM-FIQA training")
    parser.add_argument("--dataset", type=str, default="webface", choices=list(DATASETS))
    parser.add_argument("--data_root", type=str, required=True,
                        help="image folder (webface) or directory with train.rec/train.idx (ms1mv2)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--network", type=str, default=None, choices=["iresnet50", "iresnet100"],
                        help="defaults to iresnet50 for webface and iresnet100 for ms1mv2")
    parser.add_argument("--num_classes", type=int, default=None, help="defaults to the full training dataset")
    parser.add_argument("--num_epoch", type=int, default=None, help="defaults to 34 (webface) / 18 (ms1mv2)")
    parser.add_argument("--loss", type=str, default="arcface", choices=["arcface", "curricularface"])
    parser.add_argument("--target", type=str, default="cumulative", choices=["cumulative", "noncumulative"],
                        help="cumulative = CARPM-FIQA, noncumulative = single-epoch RPM (CR-FIQA-style ablation)")
    parser.add_argument("--batch_size", type=int, default=128, help="per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="for a global batch size of 512")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--s", type=float, default=64.0)
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=10.0, help="weight of the quality loss (lambda)")
    parser.add_argument("--beta", type=float, default=0.5, help="Smooth L1 beta of the quality loss")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_workers", type=int, default=16)
    cfg = parser.parse_args()
    for k, v in DATASETS[cfg.dataset].items():
        if getattr(cfg, k, None) is None:
            setattr(cfg, k, v)
    return cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def init_logging(rank, output):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        for handler in [logging.FileHandler(os.path.join(output, "training.log")), logging.StreamHandler(sys.stdout)]:
            handler.setFormatter(formatter)
            log_root.addHandler(handler)


def main(cfg):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        os.makedirs(cfg.output, exist_ok=True)
    dist.barrier()
    init_logging(rank, cfg.output)
    logging.info("Config is: {}".format(vars(cfg)))

    if cfg.dataset == "webface":
        trainset = ImageFolderDataset(cfg.data_root, num_classes=cfg.num_classes)
    else:
        trainset = MXFaceDataset(cfg.data_root)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    network = iresnet100 if cfg.network == "iresnet100" else iresnet50
    backbone = network(dropout=0.4, num_features=cfg.embedding_size, qs=1).to(local_rank)
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = DistributedDataParallel(module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    header_cls = losses.ArcFace if cfg.loss == "arcface" else losses.CurricularFace
    header = header_cls(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    header = DistributedDataParallel(module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    lr = cfg.lr / 512 * cfg.batch_size * world_size
    opt_backbone = torch.optim.SGD(params=[{"params": backbone.parameters()}], lr=lr,
                                   momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(params=[{"params": header.parameters()}], lr=lr,
                                 momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def lr_step_func(epoch):
        return 0.1 ** len([m for m in cfg.lr_milestones if m - 1 <= epoch])

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_header, lr_lambda=lr_step_func)

    criterion = CrossEntropyLoss()
    criterion_qs = torch.nn.SmoothL1Loss(beta=cfg.beta)

    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    logging.info("Total Step is: %d" % total_step)

    # rpm_history[i, n] holds Q(x_i)_n, the cumulative average of the relative point margin
    # q_j(x_i) = CCS / NNCCS over epochs j = 0..n (Eq. 3). Each rank writes the samples it sees;
    # the slots are merged across ranks at the end of every epoch.
    num_samples = len(trainset) + 1
    rpm_history = torch.zeros((num_samples, cfg.num_epoch, 1), dtype=torch.float32, device=local_rank)

    global_step = 0
    tic = time.time()
    for epoch in range(cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for img, label, idx in train_loader:
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features, qs = backbone(img)
            thetas, ccs, nnccs = header(features, label)

            rpm = ccs / (nnccs + 1e-6)
            if epoch == 0:
                rpm_history[idx, epoch] = rpm
            else:
                rpm_history[idx, epoch] = (rpm + epoch * rpm_history[idx, epoch - 1]) / (epoch + 1)

            if cfg.target == "cumulative":
                quality_target = rpm_history[idx, epoch].reshape(-1, 1)
            else:
                quality_target = rpm.reshape(-1, 1)

            loss_cls = criterion(thetas, label)
            loss_qs = criterion_qs(quality_target, qs)
            loss_v = loss_cls + cfg.alpha * loss_qs
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()
            opt_backbone.zero_grad()
            opt_header.zero_grad()

            if rank == 0 and global_step % 50 == 0:
                speed = 50 * cfg.batch_size * world_size / (time.time() - tic)
                logging.info(
                    f"Speed {speed:.2f} samples/sec   Total Loss {loss_v.item():.4f}   Cls Loss {loss_cls.item():.4f}   "
                    f"Quality Loss {loss_qs.item():.4f}   Epoch: {epoch}   Global Step: {global_step}"
                )
                tic = time.time()

        scheduler_backbone.step()
        scheduler_header.step()

        # merge the slots written by the different ranks (unseen slots are 0 on the other ranks). As in the
        # runs reported in the paper, MAX also clips negative values (CCS < 0, mostly in the first epochs) to 0.
        dist.all_reduce(rpm_history, op=dist.ReduceOp.MAX)

        if rank == 0:
            np.savez(os.path.join(cfg.output, "quality_history.npz"),
                     quality_history=rpm_history.cpu().numpy(), file_paths=np.asarray(trainset.imgidx))
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, f"{global_step}backbone.pth"))
            torch.save(header.module.state_dict(), os.path.join(cfg.output, f"{global_step}header.pth"))

    dist.destroy_process_group()


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    main(cfg)
