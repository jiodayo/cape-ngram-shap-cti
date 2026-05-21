#!/usr/bin/env python3
"""Export per-sample top-k SHAP contributions for ngram/skip/desc LGBM models.

Outputs one CSV per label with top-k features per sample.
Designed for single-label datasets (family) where positives per label cover all samples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import shap
from scipy import sparse

from common import (
    aggregate_description_features,
    build_description_corpus,
    build_feature_names,
    build_skipgram_tokens,
    extract_api_sequence,
    extract_positive_shap_values,
    get_label_names,
    iter_batches,
    load_json,
    load_split,
    resolve_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample top-k SHAP contributions for ngram/skip/desc LGBM models."
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("logs/cape_ngram_desc_full_lgbm_seed42_106"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("logs/cape_ngram_desc_full_lgbm_seed42_106/lgbm/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample"),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/avast_ctu_cape/ngram_dataset_family_104"),
    )
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--label-set", type=Path, default=None)
    parser.add_argument("--api-list", type=Path, default=None)
    parser.add_argument("--api-descriptions", type=Path, default=None)

    parser.add_argument("--max-test-files", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument("--skip-max-gap", type=int, default=2)
    parser.add_argument("--skip-window", type=int, default=4)
    parser.add_argument("--seq-dedupe-consecutive", action="store_true")
    parser.add_argument("--api-count-log1p", action="store_true")

    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--all-samples", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dir = args.train_dir or args.data_root / "train"
    test_dir = args.test_dir or args.data_root / "test"
    label_set = args.label_set or args.data_root / "label_set.json"
    api_list_path = args.api_list or args.data_root / "api.json"
    api_desc_path = args.api_descriptions or args.data_root / "api_descriptions.json"

    feature_root = args.feature_root
    seq_vectorizer = joblib.load(feature_root / "sequence_vectorizer.joblib")
    skip_vectorizer = joblib.load(feature_root / "skip_vectorizer.joblib")
    desc_vectorizer = joblib.load(
        feature_root / "description_vectorizer.joblib")

    label_to_index: Dict[str, int] = load_json(label_set)
    label_names = get_label_names(label_to_index)
    api_list: List[str] = load_json(api_list_path)
    api_descriptions: Dict[str, str] = load_json(api_desc_path)

    api_to_index = {name: i for i, name in enumerate(api_list)}

    seq_texts, skip_texts, api_counts, y_test, test_files = load_split(
        data_dir=test_dir,
        label_to_index=label_to_index,
        api_to_index=api_to_index,
        max_files=args.max_test_files,
        max_seq_len=args.max_seq_len,
        dedupe_consecutive=args.seq_dedupe_consecutive,
        use_skipgram=True,
        skip_max_gap=args.skip_max_gap,
        skip_window=args.skip_window,
        api_count_log1p=args.api_count_log1p,
    )

    x_seq = seq_vectorizer.transform(seq_texts).tocsr()
    x_skip = skip_vectorizer.transform(skip_texts).tocsr()
    desc_corpus = build_description_corpus(api_list, api_descriptions)
    api_desc_tfidf = desc_vectorizer.transform(desc_corpus).tocsr()
    x_desc = aggregate_description_features(api_counts, api_desc_tfidf)

    x_test = sparse.hstack([x_seq, x_skip, x_desc],
                           format="csr", dtype=np.float32)

    feature_names = build_feature_names(seq_vectorizer, skip_vectorizer, desc_vectorizer)

    labels = resolve_labels(args.labels, label_names, args.model_dir)
    if not labels:
        raise RuntimeError("No label models found in model-dir.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_state)

    for label in labels:
        label_idx = label_to_index.get(label)
        if label_idx is None:
            continue

        if args.all_samples:
            label_indices = np.arange(x_test.shape[0])
        else:
            label_indices = np.where(y_test[:, label_idx] == 1)[0]

        if args.max_samples > 0 and len(label_indices) > args.max_samples:
            label_indices = rng.choice(
                label_indices, size=args.max_samples, replace=False)

        model_path = args.model_dir / f"{label}.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)

        out_path = output_dir / f"shap_per_sample_topk_{label}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample",
                    "label",
                    "rank",
                    "feature",
                    "shap_value",
                    "abs_shap",
                    "feature_value",
                ]
            )

            for batch_indices in iter_batches(label_indices, args.batch_size):
                x_batch = x_test[batch_indices].toarray()
                shap_values = extract_positive_shap_values(explainer, x_batch)

                for row_offset, sample_idx in enumerate(batch_indices):
                    shap_row = shap_values[row_offset]
                    abs_row = np.abs(shap_row)

                    if args.top_k < len(abs_row):
                        top_idx = np.argpartition(-abs_row,
                                                  args.top_k - 1)[: args.top_k]
                        top_idx = top_idx[np.argsort(-abs_row[top_idx])]
                    else:
                        top_idx = np.argsort(-abs_row)

                    sample_name = test_files[sample_idx]
                    for rank, feat_idx in enumerate(top_idx, start=1):
                        writer.writerow(
                            [
                                sample_name,
                                label,
                                rank,
                                feature_names[feat_idx],
                                float(shap_row[feat_idx]),
                                float(abs_row[feat_idx]),
                                float(x_batch[row_offset, feat_idx]),
                            ]
                        )

        print(f"[INFO] per-sample SHAP saved: {out_path}")


if __name__ == "__main__":
    main()
