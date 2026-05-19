#!/usr/bin/env python3
"""Export per-sample top-k SHAP contributions for ngram/skip/desc LGBM models.

Outputs one CSV per label with top-k features per sample.
Designed for single-label datasets (family) where positives per label cover all samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import shap
from scipy import sparse


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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_label_names(label_to_index: Dict[str, int]) -> List[str]:
    num_labels = max(label_to_index.values()) + 1
    names = [""] * num_labels
    for name, index in label_to_index.items():
        names[index] = name
    for i, name in enumerate(names):
        if not name:
            names[i] = f"label_{i}"
    return names


def extract_api_sequence(
    sample: dict,
    max_len: int,
    dedupe_consecutive: bool,
) -> List[str]:
    sequence = sample.get("apicalls", [])
    if not isinstance(sequence, list) or not sequence:
        sequence = sample.get("DeduplicationApicalls", [])
    if not isinstance(sequence, list):
        sequence = []

    tokens: List[str] = []
    last_token = ""
    for token in sequence:
        if isinstance(token, str) and token:
            if dedupe_consecutive and token == last_token:
                continue
            tokens.append(token)
            last_token = token

    if max_len > 0:
        tokens = tokens[:max_len]
    return tokens


def build_skipgram_tokens(sequence: Sequence[str], max_skip: int, window_size: int) -> List[str]:
    if max_skip < 0:
        return []

    n = len(sequence)
    features: List[str] = []
    for i in range(n):
        max_j = min(n, i + window_size + 1)
        for j in range(i + 1, max_j):
            gap = j - i - 1
            if gap <= max_skip:
                features.append(f"{sequence[i]}__SKIP{gap}__{sequence[j]}")
    return features


def load_split(
    data_dir: Path,
    label_to_index: Dict[str, int],
    api_to_index: Dict[str, int],
    max_files: int,
    max_seq_len: int,
    dedupe_consecutive: bool,
    use_skipgram: bool,
    skip_max_gap: int,
    skip_window: int,
    api_count_log1p: bool,
) -> Tuple[List[str], List[str], sparse.csr_matrix, np.ndarray, List[str]]:
    json_files = sorted([p for p in data_dir.glob("*.json")])
    if max_files > 0:
        json_files = json_files[:max_files]

    num_labels = max(label_to_index.values()) + 1
    seq_texts: List[str] = []
    skip_texts: List[str] = []
    y = np.zeros((len(json_files), num_labels), dtype=np.int8)

    row_idx: List[int] = []
    col_idx: List[int] = []
    values: List[float] = []

    for i, file_path in enumerate(json_files):
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = json.load(f)

        sequence = extract_api_sequence(
            sample,
            max_len=max_seq_len,
            dedupe_consecutive=dedupe_consecutive,
        )
        seq_texts.append(" ".join(sequence))

        if use_skipgram:
            skip_tokens = build_skipgram_tokens(
                sequence, max_skip=skip_max_gap, window_size=skip_window
            )
            skip_texts.append(" ".join(skip_tokens))
        else:
            skip_texts.append("")

        sample_labels = sample.get("functions", [])
        if isinstance(sample_labels, list):
            for label in sample_labels:
                if label in label_to_index:
                    y[i, label_to_index[label]] = 1

        counts = {}
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
        for api_name, count in counts.items():
            api_index = api_to_index.get(api_name)
            if api_index is not None:
                row_idx.append(i)
                col_idx.append(api_index)
                if api_count_log1p:
                    values.append(float(math.log1p(count)))
                else:
                    values.append(float(count))

    api_count_matrix = sparse.csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(json_files), len(api_to_index)),
        dtype=np.float32,
    )

    return seq_texts, skip_texts, api_count_matrix, y, [p.name for p in json_files]


def build_description_corpus(
    api_list: List[str],
    api_descriptions: Dict[str, str],
) -> List[str]:
    corpus: List[str] = []
    for api_name in api_list:
        desc = api_descriptions.get(api_name, "")
        if not isinstance(desc, str) or not desc.strip():
            desc = api_name.replace("_", " ")
        corpus.append(desc)
    return corpus


def aggregate_description_features(
    api_count_matrix: sparse.csr_matrix,
    api_desc_tfidf: sparse.csr_matrix,
) -> sparse.csr_matrix:
    row_sum = np.asarray(api_count_matrix.sum(axis=1)
                         ).ravel().astype(np.float32)
    row_sum[row_sum == 0.0] = 1.0
    normalized_count = api_count_matrix.multiply(1.0 / row_sum[:, None])
    desc_features = normalized_count @ api_desc_tfidf
    return desc_features.tocsr().astype(np.float32)


def resolve_labels(requested: str, label_names: Sequence[str], model_dir: Path) -> List[str]:
    model_labels = {p.stem for p in model_dir.glob("*.joblib")}
    available = sorted(model_labels.intersection(label_names))
    if requested.strip():
        keep = [item.strip() for item in requested.split(",") if item.strip()]
        return [lab for lab in keep if lab in model_labels]
    return available


def extract_positive_shap_values(explainer: shap.TreeExplainer, x_np: np.ndarray) -> np.ndarray:
    raw = explainer.shap_values(x_np, check_additivity=False)
    if isinstance(raw, list):
        values = raw[1] if len(raw) > 1 else raw[0]
    else:
        values = raw

    values = np.asarray(values)
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        values = values[:, :, class_idx]
    elif values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def iter_batches(indices: np.ndarray, batch_size: int) -> Sequence[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start: start + batch_size]


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

    seq_names = list(seq_vectorizer.get_feature_names_out())
    skip_names = list(skip_vectorizer.get_feature_names_out())
    desc_names = list(desc_vectorizer.get_feature_names_out())
    feature_names = (
        [f"seq:{name}" for name in seq_names]
        + [f"skip:{name}" for name in skip_names]
        + [f"desc:{name}" for name in desc_names]
    )

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
