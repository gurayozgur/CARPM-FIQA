"""Estimate CARPM-FIQA quality scores for aligned 112x112 face images.

Expects the CR-FIQA data layout:  <data_dir>/quality_data/<dataset>/image_path_list.txt, whose
lines are image paths relative to <data_dir> (e.g. quality_data/lfw/images/00000.jpg).
Writes <output_dir>/<score_name>_<dataset>.txt with lines "<image path> <quality score>".

    python evaluation/get_quality_scores.py --model_path pretrained/CARPM-FIQA(L).pth --backbone iresnet100 \
        --data_dir data --datasets lfw,agedb_30,cfp_fp,calfw,cplfw,XQLFW,adience,IJBC --score_name CARPML
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from backbones.iresnet import iresnet50, iresnet100


def parse_args():
    parser = argparse.ArgumentParser(description="CARPM-FIQA quality score extraction")
    parser.add_argument("--model_path", type=str, required=True, help="path to the CARPM-FIQA backbone .pth")
    parser.add_argument("--backbone", type=str, default="iresnet100", choices=["iresnet50", "iresnet100"],
                        help="iresnet50 for CARPM-FIQA(S), iresnet100 for CARPM-FIQA(L)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="lfw,agedb_30,cfp_fp,calfw,cplfw,XQLFW,adience,IJBC")
    parser.add_argument("--output_dir", type=str, default="scores")
    parser.add_argument("--score_name", type=str, default="CARPML")
    parser.add_argument("--color_channel", type=str, default="BGR", choices=["BGR", "RGB"],
                        help="BGR: feed images as read by OpenCV (images prepared with the CR-FIQA scripts); "
                             "RGB: convert OpenCV's BGR to RGB first")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def load_model(model_path, backbone, device):
    network = iresnet100 if backbone == "iresnet100" else iresnet50
    model = network(num_features=512, qs=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_batch(paths, color_channel):
    images = []
    for path in paths:
        image = cv2.resize(cv2.imread(path), (112, 112))
        if color_channel == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(np.transpose(image, (2, 0, 1)))
    return np.array(images)


@torch.no_grad()
def get_quality_scores(model, image_paths, device, batch_size=64, color_channel="BGR"):
    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        imgs = torch.Tensor(load_batch(image_paths[i:i + batch_size], color_channel)).to(device)
        imgs.div_(255).sub_(0.5).div_(0.5)
        _, qs = model(imgs)
        scores.append(qs.cpu().numpy())
    return np.vstack(scores)[:, 0]


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.backbone, device)
    os.makedirs(args.output_dir, exist_ok=True)
    for dataset in args.datasets.split(","):
        with open(os.path.join(args.data_dir, "quality_data", dataset, "image_path_list.txt")) as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        print(f"{dataset}: {len(rel_paths)} images")
        scores = get_quality_scores(model, [os.path.join(args.data_dir, p) for p in rel_paths], device,
                                    args.batch_size, args.color_channel)
        with open(os.path.join(args.output_dir, f"{args.score_name}_{dataset}.txt"), "w") as f:
            for path, score in zip(rel_paths, scores):
                f.write(f"{path} {score:.15f}\n")


if __name__ == "__main__":
    main(parse_args())
