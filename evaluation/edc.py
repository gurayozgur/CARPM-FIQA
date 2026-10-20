"""Error-versus-Discard Characteristic (EDC): FNMR at fixed FMR after discarding the lowest-quality pairs.

The quality of a pair is the minimum quality of its two samples (for IJB-C: of all images of its two
templates). For every discard ratio (0%, 5%, ..., 95%) the FMR thresholds are recomputed on the
remaining pairs. Results are stored as <output_dir>/<dataset>/<method>/<fr_model>/fnmr{2,3,4}.npy
(FNMR @ FMR=1e-2/1e-3/1e-4) and unconsidered_rates.npy.

Inputs (CR-FIQA data layout, see README):
    <data_dir>/quality_data/<dataset>/pair_list.txt     "<name_a> <name_b> <is_same>"
    <embeddings_dir>/<dataset>_<fr_model>/<name>.npy     face embeddings (IJB-C: aggregated template features)
    <score_dir>/<method>_<dataset>.txt[.gz]              "<image path> <quality score>"
    <ijbc_meta_dir>/ijbc_face_tid_mid.txt, ijbc_template_pair_label.txt (IJB-C only)
"""
import argparse
import gzip
import math
import os

import numpy as np
from tqdm import tqdm

FMRS = {"fnmr2": 1e-2, "fnmr3": 1e-3, "fnmr4": 1e-4}
UNCONSIDERED_RATES = np.arange(0, 0.98, 0.05)


def parse_args():
    parser = argparse.ArgumentParser(description="EDC (FNMR vs. discard ratio) computation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--embeddings_dir", type=str, default="data/quality_embeddings")
    parser.add_argument("--score_dir", type=str, default="scores")
    parser.add_argument("--ijbc_meta_dir", type=str, default="data/IJB_release/IJBC/meta")
    parser.add_argument("--output_dir", type=str, default="results/fnmrs")
    parser.add_argument("--methods", type=str, default="CARPMS,CARPML")
    parser.add_argument("--datasets", type=str, default="adience,agedb_30,cfp_fp,lfw,calfw,cplfw,XQLFW,IJBC")
    parser.add_argument("--models", type=str, default="ArcFaceModel,ElasticFaceModel,MagFaceModel,CurricularFaceModel")
    return parser.parse_args()


def load_quality(path):
    """Returns {image name without extension: quality score}."""
    opener = gzip.open if path.endswith(".gz") else open
    quality = {}
    with opener(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                quality[os.path.splitext(os.path.basename(parts[0]))[0]] = float(parts[1])
    return quality


def find_score_file(score_dir, method, dataset):
    for name in [f"{method}_{dataset}.txt", f"{method}_{dataset}.txt.gz"]:
        if os.path.exists(os.path.join(score_dir, name)):
            return os.path.join(score_dir, name)
    raise FileNotFoundError(f"no score file for {method}/{dataset} in {score_dir}")


def threshold_at_fmr(gen_dist, imp_dist, fmr):
    """Distance threshold whose FMR is closest to the target FMR. Candidate thresholds are all
    observed comparison scores; FMR(t) = fraction of impostor similarities >= t."""
    sims = np.unique(np.concatenate([-gen_dist, -imp_dist]))
    imp_sims = np.sort(-imp_dist)
    fmr_at = (len(imp_sims) - np.searchsorted(imp_sims, sims, side="left")) / len(imp_sims)
    return -sims[np.argmin(np.abs(fmr_at - fmr))]


def edc(dist, is_same, pair_quality):
    """FNMR at each FMR in FMRS for each discard ratio in UNCONSIDERED_RATES (lowest quality discarded first)."""
    order = np.argsort(pair_quality)
    num_pairs = len(order)
    fnmrs = {k: [] for k in FMRS}
    for u_rate in UNCONSIDERED_RATES:
        keep = order[int(u_rate * num_pairs):]
        same = is_same[keep] == 1
        gen_dist, imp_dist = dist[keep][same], dist[keep][~same]
        for k, fmr in FMRS.items():
            if len(gen_dist) == 0:
                fnmrs[k].append(1.0)
                continue
            th = threshold_at_fmr(gen_dist, imp_dist, fmr)
            fnmrs[k].append(np.mean(gen_dist >= th))
    return fnmrs


def load_pairs(data_dir, dataset):
    names_a, names_b, is_same = [], [], []
    with open(os.path.join(data_dir, "quality_data", dataset, "pair_list.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                names_a.append(parts[0])
                names_b.append(parts[1])
                is_same.append(int(parts[2]))
    return names_a, names_b, np.array(is_same)


def pair_distances(emb_dir, names_a, names_b):
    """Normalized angular distance arccos(cos(a, b)) / pi."""
    cache = {n: np.load(os.path.join(emb_dir, n + ".npy")).astype(np.float64) for n in set(names_a) | set(names_b)}
    a = np.stack([cache[n] for n in names_a])
    b = np.stack([cache[n] for n in names_b])
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.arccos(np.clip(cos, -1.0, 1.0)) / math.pi


def ijbc_pairs(meta_dir):
    tid_mid = np.loadtxt(os.path.join(meta_dir, "ijbc_face_tid_mid.txt"), dtype=str)
    pairs = np.loadtxt(os.path.join(meta_dir, "ijbc_template_pair_label.txt"), dtype=np.int64)
    images = np.array([os.path.splitext(n)[0] for n in tid_mid[:, 0]])
    return images, tid_mid[:, 1].astype(np.int64), pairs[:, 0], pairs[:, 1], pairs[:, 2]


def ijbc_template_quality(images, templates, quality):
    """Template quality = minimum quality over the template's images."""
    uniq, inv = np.unique(templates, return_inverse=True)
    q = np.full(len(uniq), np.inf)
    np.minimum.at(q, inv, np.array([quality[n] for n in images]))
    return dict(zip(uniq, q))


def ijbc_pair_distances(emb_dir, t1, t2, chunk=1_000_000):
    """Squared Euclidean distance of the L2-normalized template features, computed in float16
    as in the evaluation used for the paper (CR-FIQA IJB-C protocol)."""
    from sklearn.preprocessing import normalize

    uniq, inv = np.unique(np.concatenate([t1, t2]), return_inverse=True)
    feats = np.stack([np.load(os.path.join(emb_dir, f"{t}.npy")) for t in tqdm(uniq, desc="templates")])
    feats = feats.astype(np.float16)
    i1, i2 = inv[:len(t1)], inv[len(t1):]
    dist = np.empty(len(t1), dtype=np.float16)
    for s in tqdm(range(0, len(t1), chunk), desc="distances"):
        x = normalize(feats[i1[s:s + chunk]])
        y = normalize(feats[i2[s:s + chunk]])
        dist[s:s + chunk] = np.sum(np.square(np.subtract(x, y)), 1)
    return dist


def save(fnmrs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in fnmrs.items():
        np.save(os.path.join(out_dir, f"{k}.npy"), np.array(v))
    np.save(os.path.join(out_dir, "unconsidered_rates.npy"), UNCONSIDERED_RATES)


def main(args):
    methods, models = args.methods.split(","), args.models.split(",")
    for dataset in args.datasets.split(","):
        qualities = {m: load_quality(find_score_file(args.score_dir, m, dataset)) for m in methods}
        if dataset == "IJBC":
            images, templates, t1, t2, is_same = ijbc_pairs(args.ijbc_meta_dir)
            pair_quality = {}
            for m in methods:
                tq = ijbc_template_quality(images, templates, qualities[m])
                pair_quality[m] = np.minimum(np.array([tq[t] for t in t1]), np.array([tq[t] for t in t2]))
        else:
            names_a, names_b, is_same = load_pairs(args.data_dir, dataset)
            pair_quality = {m: np.minimum(np.array([qualities[m][n] for n in names_a]),
                                          np.array([qualities[m][n] for n in names_b])) for m in methods}
        for model in models:
            emb_dir = os.path.join(args.embeddings_dir, f"{dataset}_{model}")
            if dataset == "IJBC":
                dist = ijbc_pair_distances(emb_dir, t1, t2)
            else:
                dist = pair_distances(emb_dir, names_a, names_b)
            for m in methods:
                fnmrs = edc(dist, is_same, pair_quality[m])
                save(fnmrs, os.path.join(args.output_dir, dataset, m, model))
                print(f"{dataset:9s} {model:20s} {m:8s} FNMR@1e-3: {fnmrs['fnmr3'][0]:.4f} -> {fnmrs['fnmr3'][6]:.4f} "
                      f"(30% discarded)")


if __name__ == "__main__":
    main(parse_args())
