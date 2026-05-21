#!/usr/bin/env python3
"""Export per-sample SHAP waterfall/force plots for ngram/skip/desc LGBM models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
import matplotlib.pyplot as plt
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
    shorten_feature_name,
)

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample SHAP waterfall/force plots for ngram/skip/desc LGBM models."
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
            "shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample_plots"),
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

    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument("--skip-max-gap", type=int, default=2)
    parser.add_argument("--skip-window", type=int, default=4)
    parser.add_argument("--seq-dedupe-consecutive", action="store_true")
    parser.add_argument("--api-count-log1p", action="store_true")

    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--all-samples", action="store_true")

    parser.add_argument(
        "--output-type",
        type=str,
        default="both",
        choices=["waterfall", "force", "both"],
    )
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--max-feature-name-len", type=int, default=32)
    parser.add_argument("--feature-name-tail-len", type=int, default=8)

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

    seq_texts_all: List[str] = []
    skip_texts_all: List[str] = []
    api_counts_all: List[sparse.csr_matrix] = []
    y_all: List[np.ndarray] = []
    sample_names_all: List[str] = []

    if args.split in ("train", "all"):
        seq_texts, skip_texts, api_counts, y_split, sample_names = load_split(
            data_dir=train_dir,
            label_to_index=label_to_index,
            api_to_index=api_to_index,
            max_files=0,
            max_seq_len=args.max_seq_len,
            dedupe_consecutive=args.seq_dedupe_consecutive,
            use_skipgram=True,
            skip_max_gap=args.skip_max_gap,
            skip_window=args.skip_window,
            api_count_log1p=args.api_count_log1p,
            name_prefix="train_" if args.split == "all" else "",
        )
        seq_texts_all.extend(seq_texts)
        skip_texts_all.extend(skip_texts)
        api_counts_all.append(api_counts)
        y_all.append(y_split)
        sample_names_all.extend(sample_names)

    if args.split in ("test", "all"):
        seq_texts, skip_texts, api_counts, y_split, sample_names = load_split(
            data_dir=test_dir,
            label_to_index=label_to_index,
            api_to_index=api_to_index,
            max_files=0,
            max_seq_len=args.max_seq_len,
            dedupe_consecutive=args.seq_dedupe_consecutive,
            use_skipgram=True,
            skip_max_gap=args.skip_max_gap,
            skip_window=args.skip_window,
            api_count_log1p=args.api_count_log1p,
            name_prefix="test_" if args.split == "all" else "",
        )
        seq_texts_all.extend(seq_texts)
        skip_texts_all.extend(skip_texts)
        api_counts_all.append(api_counts)
        y_all.append(y_split)
        sample_names_all.extend(sample_names)

    if not seq_texts_all:
        raise RuntimeError("No samples loaded. Check split and data paths.")

    api_counts = sparse.vstack(api_counts_all, format="csr")
    y = np.vstack(y_all)

    x_seq = seq_vectorizer.transform(seq_texts_all).tocsr()
    x_skip = skip_vectorizer.transform(skip_texts_all).tocsr()
    desc_corpus = build_description_corpus(api_list, api_descriptions)
    api_desc_tfidf = desc_vectorizer.transform(desc_corpus).tocsr()
    x_desc = aggregate_description_features(api_counts, api_desc_tfidf)

    x_all = sparse.hstack([x_seq, x_skip, x_desc],
                          format="csr", dtype=np.float32)

    feature_names = build_feature_names(seq_vectorizer, skip_vectorizer, desc_vectorizer)
    display_feature_names = [
        shorten_feature_name(
            name, args.max_feature_name_len, args.feature_name_tail_len
        )
        for name in feature_names
    ]

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
            label_indices = np.arange(x_all.shape[0])
        else:
            label_indices = np.where(y[:, label_idx] == 1)[0]

        if args.max_samples > 0 and len(label_indices) > args.max_samples:
            label_indices = rng.choice(
                label_indices, size=args.max_samples, replace=False)

        model_path = args.model_dir / f"{label}.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1] if len(
                base_value) > 1 else base_value[0])
        else:
            base_value = float(base_value)

        label_dir = output_dir / label
        waterfall_dir = label_dir / "waterfall"
        force_dir = label_dir / "force"
        label_dir.mkdir(parents=True, exist_ok=True)
        if args.output_type in ("waterfall", "both"):
            waterfall_dir.mkdir(parents=True, exist_ok=True)
        if args.output_type in ("force", "both"):
            force_dir.mkdir(parents=True, exist_ok=True)

        index_path = label_dir / "index.csv"
        with index_path.open("w", encoding="utf-8", newline="") as f:
            f.write("sample,waterfall_path,force_path\n")

            for batch_indices in iter_batches(label_indices, args.batch_size):
                x_batch = x_all[batch_indices].toarray()
                shap_values = extract_positive_shap_values(explainer, x_batch)

                for row_offset, sample_idx in enumerate(batch_indices):
                    sample_name = sample_names_all[sample_idx]
                    shap_row = shap_values[row_offset]
                    sample_features = x_batch[row_offset]

                    explanation = shap.Explanation(
                        values=shap_row,
                        base_values=base_value,
                        data=sample_features,
                        feature_names=display_feature_names,
                    )

                    waterfall_path = ""
                    force_path = ""

                    if args.output_type in ("waterfall", "both"):
                        waterfall_path = str(
                            waterfall_dir / f"shap_waterfall_{sample_name}.png")
                        plt.figure()
                        shap.plots.waterfall(
                            explanation, max_display=args.max_display, show=False
                        )
                        plt.tight_layout()
                        plt.savefig(waterfall_path, dpi=150,
                                    bbox_inches="tight")
                        plt.close()

                    if args.output_type in ("force", "both"):
                        force_path = str(
                            force_dir / f"shap_force_{sample_name}.html")
                        force_plot = shap.force_plot(
                            base_value,
                            shap_row,
                            sample_features,
                            feature_names=display_feature_names,
                            matplotlib=False,
                        )
                        shap.save_html(force_path, force_plot)

                    f.write(f"{sample_name},{waterfall_path},{force_path}\n")

        print(f"[INFO] {label}: {len(label_indices)} samples processed.")


if __name__ == "__main__":
    main()
