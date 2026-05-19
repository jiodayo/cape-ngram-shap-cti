#!/usr/bin/env python3
"""
Binary Relevance training with sequence n-grams + skip-grams + API description features.

Main ideas:
1. Sequence features: token n-grams from ordered API calls.
2. Optional skip-gram features: capture short-range non-adjacent transitions.
3. Description features: aggregate API description TF-IDF vectors by sample API frequency.

This script supports LightGBM and RandomForest (with SVD compression for sparse input).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


def resolve_existing_path(*candidates: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


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
    dedupe_consecutive: bool = False,
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

    label_names = get_label_names(label_to_index)
    num_labels = len(label_names)

    seq_texts: List[str] = []
    skip_texts: List[str] = []
    y = np.zeros((len(json_files), num_labels), dtype=np.int8)

    row_idx: List[int] = []
    col_idx: List[int] = []
    values: List[float] = []

    for i, file_path in enumerate(tqdm(json_files, desc=f"Loading {data_dir.name}")):
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
                sequence, max_skip=skip_max_gap, window_size=skip_window)
            skip_texts.append(" ".join(skip_tokens))
        else:
            skip_texts.append("")

        sample_labels = sample.get("functions", [])
        if isinstance(sample_labels, list):
            for label in sample_labels:
                if label in label_to_index:
                    y[i, label_to_index[label]] = 1

        counts = Counter(sequence)
        meta_tokens = sample.get("meta_tokens", [])
        if isinstance(meta_tokens, list):
            for token in meta_tokens:
                if isinstance(token, str) and token:
                    counts[token] += 1
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


def build_description_matrix(
    api_list: List[str],
    api_descriptions: Dict[str, str],
    max_features: int,
) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    corpus = build_description_corpus(api_list, api_descriptions)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        lowercase=True,
        dtype=np.float32,
    )
    api_desc_tfidf = vectorizer.fit_transform(corpus).tocsr()
    return vectorizer, api_desc_tfidf


def build_description_embedding_matrix(
    api_list: List[str],
    api_descriptions: Dict[str, str],
    model_name: str,
    normalize_embeddings: bool,
    batch_size: int,
) -> np.ndarray:
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers is not installed. Install it with: pip install sentence-transformers"
        )

    corpus = build_description_corpus(api_list, api_descriptions)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)


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


def aggregate_description_embedding_features(
    api_count_matrix: sparse.csr_matrix,
    api_desc_embeddings: np.ndarray,
) -> np.ndarray:
    row_sum = np.asarray(api_count_matrix.sum(axis=1)
                         ).ravel().astype(np.float32)
    row_sum[row_sum == 0.0] = 1.0

    normalized_count = api_count_matrix.multiply(1.0 / row_sum[:, None])
    desc_features = normalized_count @ api_desc_embeddings
    if sparse.issparse(desc_features):
        desc_features = desc_features.toarray()
    return np.asarray(desc_features, dtype=np.float32)


def vectorize_sequence_features(
    train_texts: List[str],
    test_texts: List[str],
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_features: int,
    use_hash: bool,
    hash_n_features: int,
    hash_binary: bool,
    hash_alternate_sign: bool,
) -> Tuple[CountVectorizer, sparse.csr_matrix, sparse.csr_matrix]:
    if use_hash:
        vectorizer = HashingVectorizer(
            ngram_range=(ngram_min, ngram_max),
            n_features=hash_n_features,
            binary=hash_binary,
            alternate_sign=hash_alternate_sign,
            norm=None,
            token_pattern=r"(?u)\b[\w_]+\b",
            dtype=np.float32,
        )
        x_train = vectorizer.transform(train_texts).tocsr()
        x_test = vectorizer.transform(test_texts).tocsr()
    else:
        vectorizer = CountVectorizer(
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
            token_pattern=r"(?u)\b[\w_]+\b",
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(train_texts).tocsr()
        x_test = vectorizer.transform(test_texts).tocsr()
    return vectorizer, x_train, x_test


def prepare_features(
    train_seq_texts: List[str],
    test_seq_texts: List[str],
    train_skip_texts: List[str],
    test_skip_texts: List[str],
    train_api_counts: sparse.csr_matrix,
    test_api_counts: sparse.csr_matrix,
    api_desc_tfidf: sparse.csr_matrix | None,
    api_desc_embeddings: np.ndarray | None,
    args: argparse.Namespace,
):
    print("Vectorizing sequence n-gram features...")
    seq_vectorizer, x_seq_train, x_seq_test = vectorize_sequence_features(
        train_texts=train_seq_texts,
        test_texts=test_seq_texts,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.seq_min_df,
        max_features=args.max_seq_features,
        use_hash=args.seq_hash,
        hash_n_features=args.hash_n_features,
        hash_binary=args.hash_binary,
        hash_alternate_sign=args.hash_alternate_sign,
    )

    x_parts_train = [x_seq_train]
    x_parts_test = [x_seq_test]

    skip_vectorizer = None
    if args.use_skipgram:
        print("Vectorizing skip-gram features...")
        skip_vectorizer, x_skip_train, x_skip_test = vectorize_sequence_features(
            train_texts=train_skip_texts,
            test_texts=test_skip_texts,
            ngram_min=1,
            ngram_max=1,
            min_df=args.skip_min_df,
            max_features=args.max_skip_features,
            use_hash=args.skip_hash,
            hash_n_features=args.hash_n_features,
            hash_binary=args.hash_binary,
            hash_alternate_sign=args.hash_alternate_sign,
        )
        x_parts_train.append(x_skip_train)
        x_parts_test.append(x_skip_test)

    desc_tfidf_features = 0
    desc_embedding_features = 0

    if args.desc_mode in {"tfidf", "hybrid"}:
        if api_desc_tfidf is None:
            raise RuntimeError(
                "Description TF-IDF mode requested but TF-IDF matrix is missing.")
        print("Aggregating API description TF-IDF features...")
        x_desc_tfidf_train = aggregate_description_features(
            train_api_counts, api_desc_tfidf)
        x_desc_tfidf_test = aggregate_description_features(
            test_api_counts, api_desc_tfidf)
        x_parts_train.append(x_desc_tfidf_train)
        x_parts_test.append(x_desc_tfidf_test)
        desc_tfidf_features = int(x_desc_tfidf_train.shape[1])

    if args.desc_mode in {"embedding", "hybrid"}:
        if api_desc_embeddings is None:
            raise RuntimeError(
                "Description embedding mode requested but embedding matrix is missing.")
        print("Aggregating API description embedding features...")
        x_desc_emb_train = aggregate_description_embedding_features(
            train_api_counts, api_desc_embeddings)
        x_desc_emb_test = aggregate_description_embedding_features(
            test_api_counts, api_desc_embeddings)
        x_desc_emb_train_sparse = sparse.csr_matrix(
            x_desc_emb_train, dtype=np.float32)
        x_desc_emb_test_sparse = sparse.csr_matrix(
            x_desc_emb_test, dtype=np.float32)
        x_parts_train.append(x_desc_emb_train_sparse)
        x_parts_test.append(x_desc_emb_test_sparse)
        desc_embedding_features = int(x_desc_emb_train_sparse.shape[1])

    x_train = sparse.hstack(x_parts_train, format="csr", dtype=np.float32)
    x_test = sparse.hstack(x_parts_test, format="csr", dtype=np.float32)

    feature_info = {
        "sequence_ngram_features": int(x_seq_train.shape[1]),
        "skipgram_features": int(x_parts_train[1].shape[1]) if args.use_skipgram else 0,
        "description_tfidf_features": desc_tfidf_features,
        "description_embedding_features": desc_embedding_features,
        "description_features": int(desc_tfidf_features + desc_embedding_features),
        "desc_mode": args.desc_mode,
        "seq_hash": bool(args.seq_hash),
        "skip_hash": bool(args.skip_hash),
        "hash_n_features": int(args.hash_n_features),
        "hash_binary": bool(args.hash_binary),
        "hash_alternate_sign": bool(args.hash_alternate_sign),
        "seq_dedupe_consecutive": bool(args.seq_dedupe_consecutive),
        "api_count_log1p": bool(args.api_count_log1p),
        "total_features": int(x_train.shape[1]),
        "num_train_samples": int(x_train.shape[0]),
        "num_test_samples": int(x_test.shape[0]),
    }

    if args.desc_mode in {"embedding", "hybrid"}:
        feature_info["desc_embedding_model"] = args.desc_embedding_model
        feature_info["desc_embedding_normalize"] = bool(
            args.desc_embedding_normalize)

    artifacts = {
        "seq_vectorizer": seq_vectorizer,
        "skip_vectorizer": skip_vectorizer,
    }

    return x_train, x_test, feature_info, artifacts


def maybe_apply_svd_for_rf(
    x_train: sparse.csr_matrix,
    x_test: sparse.csr_matrix,
    n_components: int,
    random_state: int,
):
    if n_components <= 0:
        return x_train.toarray(), x_test.toarray(), None

    max_components = min(x_train.shape[0] - 1, x_train.shape[1] - 1)
    if max_components <= 1:
        return x_train.toarray(), x_test.toarray(), None

    n_components = min(n_components, max_components)
    print(f"Applying TruncatedSVD for RF: n_components={n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    x_train_dense = svd.fit_transform(x_train)
    x_test_dense = svd.transform(x_test)
    return x_train_dense.astype(np.float32), x_test_dense.astype(np.float32), svd


def build_estimator(model_name: str, args: argparse.Namespace):
    if model_name == "lgbm":
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )
        return LGBMClassifier(
            objective="binary",
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )

    if model_name == "rf":
        max_depth = args.rf_max_depth if args.rf_max_depth > 0 else None
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_binary_relevance(
    model_name: str,
    x_train,
    y_train: np.ndarray,
    x_test,
    y_test: np.ndarray,
    label_names: List[str],
    args: argparse.Namespace,
    output_dir: Path,
):
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_pred_all = np.zeros_like(y_test, dtype=np.int8)
    summary_rows: List[dict] = []

    for label_index, label_name in enumerate(tqdm(label_names, desc=f"Train {model_name}")):
        y_train_label = y_train[:, label_index]
        y_test_label = y_test[:, label_index]

        unique_values = np.unique(y_train_label)
        if unique_values.size < 2:
            constant_value = int(
                unique_values[0]) if unique_values.size == 1 else 0
            y_pred_label = np.full_like(
                y_test_label, fill_value=constant_value)
            y_pred_all[:, label_index] = y_pred_label

            p, r, f1, _ = precision_recall_fscore_support(
                y_test_label,
                y_pred_label,
                average="binary",
                zero_division=0,
            )
            acc = accuracy_score(y_test_label, y_pred_label)

            summary_rows.append(
                {
                    "label": label_name,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "accuracy": float(acc),
                    "train_positive": int(y_train_label.sum()),
                    "test_positive": int(y_test_label.sum()),
                    "note": "single_class_train",
                }
            )

            report_txt = classification_report(
                y_test_label,
                y_pred_label,
                target_names=[f"not_{label_name}", label_name],
                labels=[0, 1],
                zero_division=0,
            )
            (reports_dir /
             f"{label_name}_report.txt").write_text(report_txt, encoding="utf-8")
            continue

        estimator = build_estimator(model_name, args)
        estimator.fit(x_train, y_train_label)

        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(x_test)[:, 1]
            y_pred_label = (y_score >= args.threshold).astype(np.int8)
        else:
            y_pred_label = estimator.predict(x_test).astype(np.int8)

        y_pred_all[:, label_index] = y_pred_label

        p, r, f1, _ = precision_recall_fscore_support(
            y_test_label,
            y_pred_label,
            average="binary",
            zero_division=0,
        )
        acc = accuracy_score(y_test_label, y_pred_label)

        summary_rows.append(
            {
                "label": label_name,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "accuracy": float(acc),
                "train_positive": int(y_train_label.sum()),
                "test_positive": int(y_test_label.sum()),
                "note": "",
            }
        )

        model_path = models_dir / f"{label_name}.joblib"
        joblib.dump(estimator, model_path)

        report_txt = classification_report(
            y_test_label,
            y_pred_label,
            target_names=[f"not_{label_name}", label_name],
            labels=[0, 1],
            zero_division=0,
        )
        (reports_dir /
         f"{label_name}_report.txt").write_text(report_txt, encoding="utf-8")

    overall_report = classification_report(
        y_test,
        y_pred_all,
        target_names=label_names,
        zero_division=0,
    )

    micro_f1 = f1_score(y_test, y_pred_all, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred_all, average="macro", zero_division=0)
    samples_f1 = f1_score(y_test, y_pred_all,
                          average="samples", zero_division=0)

    overall_payload = {
        "model": model_name,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "samples_f1": float(samples_f1),
        "num_labels": int(y_test.shape[1]),
        "num_test_samples": int(y_test.shape[0]),
    }

    accuracy_lines = [
        "",
        "per-label accuracy",
        f"{'label':<24} {'accuracy':>9}",
    ]
    for label_index, label_name in enumerate(label_names):
        acc = accuracy_score(y_test[:, label_index],
                             y_pred_all[:, label_index])
        accuracy_lines.append(f"{label_name:<24} {acc:>9.4f}")

    write_json(output_dir / "metrics_overall.json", overall_payload)
    overall_text = overall_report.rstrip() + "\n" + "\n".join(accuracy_lines) + "\n"
    (output_dir / "overall_report.txt").write_text(overall_text, encoding="utf-8")

    csv_path = output_dir / "per_label_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "train_positive",
                "test_positive",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return overall_payload, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BR model with n-gram + API description features",
    )

    parser.add_argument("--train-dir", type=str,
                        default="2024/Dataset_Extract/2016")
    parser.add_argument("--test-dir", type=str,
                        default="2024/Dataset_Extract/2017")
    parser.add_argument(
        "--label-set",
        type=str,
        default=str(resolve_existing_path(
            "data/label_set.json", "label_set.json")),
    )
    parser.add_argument(
        "--api-list",
        type=str,
        default=str(resolve_existing_path("data/api.json", "api.json")),
    )
    parser.add_argument(
        "--api-descriptions",
        type=str,
        default=str(resolve_existing_path(
            "data/api_descriptions.json", "api_descriptions.json")),
    )
    parser.add_argument("--output-root", type=str, default="logs/ngram_desc")

    parser.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    parser.add_argument(
        "--compare-rf",
        action="store_true",
        help="When model=lgbm, also train RF baseline with the same features",
    )

    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-test-files", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument(
        "--seq-dedupe-consecutive",
        action="store_true",
        help="Collapse consecutive duplicate API calls before vectorization.",
    )
    parser.add_argument(
        "--seq-hash",
        action="store_true",
        help="Use feature hashing for sequence n-grams.",
    )

    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--seq-min-df", type=int, default=2)
    parser.add_argument("--max-seq-features", type=int, default=200000)

    parser.add_argument("--use-skipgram", action="store_true")
    parser.add_argument("--skip-max-gap", type=int, default=2)
    parser.add_argument("--skip-window", type=int, default=4)
    parser.add_argument("--skip-min-df", type=int, default=2)
    parser.add_argument("--max-skip-features", type=int, default=100000)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Use feature hashing for skip-grams.",
    )

    parser.add_argument(
        "--hash-n-features",
        type=int,
        default=2**20,
        help="Feature hashing dimension (power of two recommended).",
    )
    parser.add_argument(
        "--hash-binary",
        action="store_true",
        help="Use binary weighting for hashed features.",
    )
    parser.add_argument(
        "--hash-alternate-sign",
        action="store_true",
        help="Use signed hashing to reduce collision bias.",
    )

    parser.add_argument(
        "--api-count-log1p",
        action="store_true",
        help="Apply log1p to API counts before description aggregation.",
    )

    parser.add_argument("--desc-max-features", type=int, default=4096)
    parser.add_argument(
        "--desc-mode",
        choices=["tfidf", "embedding", "hybrid"],
        default="tfidf",
        help="Description feature mode: tfidf, embedding, or hybrid",
    )
    parser.add_argument(
        "--desc-embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model for description embeddings",
    )
    parser.add_argument(
        "--desc-embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for description embedding encoding",
    )
    parser.add_argument(
        "--desc-embedding-normalize",
        action="store_true",
        help="Normalize description embeddings to unit vectors",
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--lgbm-n-estimators", type=int, default=400)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample", type=float, default=0.9)

    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=0)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-svd-components", type=int, default=512)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_to_index: Dict[str, int] = load_json(Path(args.label_set))
    api_list: List[str] = load_json(Path(args.api_list))
    api_descriptions: Dict[str, str] = load_json(Path(args.api_descriptions))

    label_names = get_label_names(label_to_index)
    api_to_index = {name: i for i, name in enumerate(api_list)}

    print("Loading training split...")
    train_seq_texts, train_skip_texts, train_api_counts, y_train, train_files = load_split(
        data_dir=train_dir,
        label_to_index=label_to_index,
        api_to_index=api_to_index,
        max_files=args.max_train_files,
        max_seq_len=args.max_seq_len,
        dedupe_consecutive=args.seq_dedupe_consecutive,
        use_skipgram=args.use_skipgram,
        skip_max_gap=args.skip_max_gap,
        skip_window=args.skip_window,
        api_count_log1p=args.api_count_log1p,
    )

    print("Loading test split...")
    test_seq_texts, test_skip_texts, test_api_counts, y_test, test_files = load_split(
        data_dir=test_dir,
        label_to_index=label_to_index,
        api_to_index=api_to_index,
        max_files=args.max_test_files,
        max_seq_len=args.max_seq_len,
        dedupe_consecutive=args.seq_dedupe_consecutive,
        use_skipgram=args.use_skipgram,
        skip_max_gap=args.skip_max_gap,
        skip_window=args.skip_window,
        api_count_log1p=args.api_count_log1p,
    )

    if len(train_files) == 0 or len(test_files) == 0:
        raise RuntimeError(
            "No JSON samples found. Check --train-dir / --test-dir and file patterns."
        )

    desc_vectorizer = None
    api_desc_tfidf = None
    api_desc_embeddings = None

    if args.desc_mode in {"tfidf", "hybrid"}:
        print("Building API description TF-IDF matrix...")
        desc_vectorizer, api_desc_tfidf = build_description_matrix(
            api_list=api_list,
            api_descriptions=api_descriptions,
            max_features=args.desc_max_features,
        )

    if args.desc_mode in {"embedding", "hybrid"}:
        print("Building API description embedding matrix...")
        api_desc_embeddings = build_description_embedding_matrix(
            api_list=api_list,
            api_descriptions=api_descriptions,
            model_name=args.desc_embedding_model,
            normalize_embeddings=args.desc_embedding_normalize,
            batch_size=args.desc_embedding_batch_size,
        )

    x_train, x_test, feature_info, vectorizers = prepare_features(
        train_seq_texts=train_seq_texts,
        test_seq_texts=test_seq_texts,
        train_skip_texts=train_skip_texts,
        test_skip_texts=test_skip_texts,
        train_api_counts=train_api_counts,
        test_api_counts=test_api_counts,
        api_desc_tfidf=api_desc_tfidf,
        api_desc_embeddings=api_desc_embeddings,
        args=args,
    )

    print("Feature summary:")
    for key, value in feature_info.items():
        print(f"  {key}: {value}")

    write_json(output_root / "feature_summary.json", feature_info)
    write_json(
        output_root / "split_files.json",
        {
            "train_files": train_files,
            "test_files": test_files,
            "num_train_files": len(train_files),
            "num_test_files": len(test_files),
        },
    )

    joblib.dump(vectorizers["seq_vectorizer"],
                output_root / "sequence_vectorizer.joblib")
    if vectorizers["skip_vectorizer"] is not None:
        joblib.dump(vectorizers["skip_vectorizer"],
                    output_root / "skip_vectorizer.joblib")
    if desc_vectorizer is not None:
        joblib.dump(desc_vectorizer, output_root /
                    "description_vectorizer.joblib")
    if api_desc_embeddings is not None:
        np.save(output_root / "api_description_embeddings.npy",
                api_desc_embeddings)

    model_queue: List[str]
    if args.model == "lgbm" and args.compare_rf:
        model_queue = ["lgbm", "rf"]
    else:
        model_queue = [args.model]

    comparison_rows: List[dict] = []

    for model_name in model_queue:
        model_output_dir = output_root / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        x_train_model = x_train
        x_test_model = x_test
        svd_model = None

        if model_name == "rf":
            x_train_model, x_test_model, svd_model = maybe_apply_svd_for_rf(
                x_train=x_train,
                x_test=x_test,
                n_components=args.rf_svd_components,
                random_state=args.random_state,
            )
            if svd_model is not None:
                joblib.dump(svd_model, model_output_dir / "rf_svd.joblib")

        print(f"Training model: {model_name}")
        overall_payload, _ = run_binary_relevance(
            model_name=model_name,
            x_train=x_train_model,
            y_train=y_train,
            x_test=x_test_model,
            y_test=y_test,
            label_names=label_names,
            args=args,
            output_dir=model_output_dir,
        )
        comparison_rows.append(overall_payload)

    if len(comparison_rows) > 1:
        compare_path = output_root / "model_comparison.csv"
        with compare_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "micro_f1", "macro_f1",
                            "samples_f1", "num_labels", "num_test_samples"],
            )
            writer.writeheader()
            writer.writerows(comparison_rows)

    print("Done.")


if __name__ == "__main__":
    main()
