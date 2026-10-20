"""AUC-EDC and pAUC-EDC (discard ratio <= 30%) tables from the EDC curves written by edc.py.

Values are multiplied by 1000 as in the paper. An "Average" block (mean over FR models) is added, and,
when several methods are given, the "Avg. norm." column: every benchmark-FMR column is divided by the
best (lowest) method of the same FR-model block, then averaged over all benchmarks and both FMRs.
"""
import argparse
import os

import numpy as np
import pandas as pd

DATASETS = {"adience": "Adience", "agedb_30": "AgeDB-30", "cfp_fp": "CFP-FP", "lfw": "LFW", "calfw": "CALFW",
            "cplfw": "CPLFW", "XQLFW": "XQLFW", "IJBC": "IJB-C"}
FMRS = {"fnmr3": "1e-3", "fnmr4": "1e-4"}
METRICS = {"AUC": 1.0, "pAUC": 0.3}


def parse_args():
    parser = argparse.ArgumentParser(description="AUC-EDC / pAUC-EDC tables")
    parser.add_argument("--fnmr_dir", type=str, default="results/fnmrs")
    parser.add_argument("--output_dir", type=str, default="results/aucs")
    parser.add_argument("--methods", type=str, default="CARPMS,CARPML")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--models", type=str, default="ArcFaceModel,ElasticFaceModel,MagFaceModel,CurricularFaceModel")
    return parser.parse_args()


def edc_auc(fnmr, unconsidered_rates, max_rate=1.0):
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    if max_rate < 1.0:
        p_idx = np.argmin(np.abs(unconsidered_rates - max_rate)) + 1
        fnmr, unconsidered_rates = fnmr[:p_idx], unconsidered_rates[:p_idx]
    return trapezoid(fnmr, unconsidered_rates) * 1000.0


def build_table(fnmr_dir, methods, datasets, models, max_rate):
    rows = []
    for model in models:
        for method in methods:
            row = {"FR_Model": model, "Method": method}
            for dataset in datasets:
                for fnmr_file, fmr in FMRS.items():
                    path = os.path.join(fnmr_dir, dataset, method, model)
                    try:
                        fnmr = np.load(os.path.join(path, f"{fnmr_file}.npy"))
                        rates = np.load(os.path.join(path, "unconsidered_rates.npy"))
                        row[f"{DATASETS[dataset]}_{fmr}"] = edc_auc(fnmr, rates, max_rate)
                    except FileNotFoundError:
                        row[f"{DATASETS[dataset]}_{fmr}"] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    average = df.drop(columns="FR_Model").groupby("Method", sort=False).mean().reset_index()
    average.insert(0, "FR_Model", "Average")
    df = pd.concat([df, average], ignore_index=True)
    if len(methods) > 1:
        value_cols = [c for c in df.columns if c not in ("FR_Model", "Method")]
        best = df.groupby("FR_Model")[value_cols].transform("min")
        df["Avg_norm"] = (df[value_cols] / best).mean(axis=1)
    return df


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.precision", 4)
    for metric, max_rate in METRICS.items():
        df = build_table(args.fnmr_dir, args.methods.split(","), args.datasets.split(","), args.models.split(","),
                         max_rate)
        df.to_csv(os.path.join(args.output_dir, f"{metric}_results.csv"), index=False, float_format="%.4f")
        print(f"\n{metric}-EDC (x1000)\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main(parse_args())
