#!/usr/bin/env python3
"""Common utilities shared across SHAP analysis and CTI scripts.

This module consolidates functions that were previously duplicated across:
- analyze_shap_ngram_desc_lgbm_group.py
- export_shap_ngram_desc_lgbm_per_sample.py
- export_shap_ngram_desc_lgbm_per_sample_plots.py
- cti_attach_shap_explanations.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Default paths configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_paths.json"


def load_default_paths() -> Dict[str, str]:
    """Load default paths from config/default_paths.json if it exists."""
    if _DEFAULT_CONFIG_PATH.exists():
        return load_json(_DEFAULT_CONFIG_PATH)
    return {}


def resolve_default(value, key: str, fallback: str = "") -> str:
    """Return *value* if truthy, else look up *key* in default_paths, else *fallback*."""
    if value:
        return str(value)
    defaults = load_default_paths()
    result = defaults.get(key, fallback)
    # Support {feature_root} style interpolation
    if "{" in result:
        for k, v in defaults.items():
            result = result.replace("{" + k + "}", v)
    return result


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_json(path: Path):
    """Load a JSON file and return the parsed object."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def get_label_names(label_to_index: Dict[str, int]) -> List[str]:
    """Convert label-to-index dict into an ordered list of label names."""
    num_labels = max(label_to_index.values()) + 1
    names = [""] * num_labels
    for name, index in label_to_index.items():
        names[index] = name
    for i, name in enumerate(names):
        if not name:
            names[i] = f"label_{i}"
    return names


def resolve_labels(
    requested: str,
    label_names: Sequence[str],
    model_dir: Path,
) -> List[str]:
    """Resolve which labels to process based on available models."""
    model_labels = {p.stem for p in model_dir.glob("*.joblib")}
    available = sorted(model_labels.intersection(label_names))
    if requested.strip():
        keep = [item.strip() for item in requested.split(",") if item.strip()]
        return [lab for lab in keep if lab in model_labels]
    return available


# ---------------------------------------------------------------------------
# API sequence extraction
# ---------------------------------------------------------------------------

def extract_api_sequence(
    sample: dict,
    max_len: int,
    dedupe_consecutive: bool = False,
) -> List[str]:
    """Extract ordered API call sequence from a sample dict."""
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


# ---------------------------------------------------------------------------
# Skip-gram features
# ---------------------------------------------------------------------------

def build_skipgram_tokens(
    sequence: Sequence[str],
    max_skip: int,
    window_size: int,
) -> List[str]:
    """Build skip-gram feature tokens from an API call sequence."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    name_prefix: str = "",
) -> Tuple[List[str], List[str], sparse.csr_matrix, np.ndarray, List[str]]:
    """Load a data split directory and return text features, API counts, and labels.

    Parameters
    ----------
    name_prefix : str
        Optional prefix for sample names (e.g. "train_" or "test_").
    """
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
        meta_tokens = sample.get("meta_tokens", [])
        if isinstance(meta_tokens, list):
            for token in meta_tokens:
                if isinstance(token, str) and token:
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

    sample_names = [f"{name_prefix}{p.name}" for p in json_files]
    return seq_texts, skip_texts, api_count_matrix, y, sample_names


# ---------------------------------------------------------------------------
# Description features
# ---------------------------------------------------------------------------

def build_description_corpus(
    api_list: List[str],
    api_descriptions: Dict[str, str],
) -> List[str]:
    """Build a text corpus from API descriptions for TF-IDF vectorization."""
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
    """Aggregate API description TF-IDF vectors weighted by API frequency."""
    row_sum = np.asarray(api_count_matrix.sum(axis=1)
                         ).ravel().astype(np.float32)
    row_sum[row_sum == 0.0] = 1.0
    normalized_count = api_count_matrix.multiply(1.0 / row_sum[:, None])
    desc_features = normalized_count @ api_desc_tfidf
    return desc_features.tocsr().astype(np.float32)


# ---------------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------------

def extract_positive_shap_values(explainer, x_np: np.ndarray) -> np.ndarray:
    """Extract SHAP values for the positive class from a TreeExplainer."""
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


def iter_batches(indices: np.ndarray, batch_size: int):
    """Yield successive batch slices of indices."""
    for start in range(0, len(indices), batch_size):
        yield indices[start: start + batch_size]


# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------

def build_feature_names(
    seq_vectorizer,
    skip_vectorizer=None,
    desc_vectorizer=None,
) -> List[str]:
    """Build a unified feature name list with group prefixes."""
    seq_names = list(seq_vectorizer.get_feature_names_out())
    skip_names = (
        list(skip_vectorizer.get_feature_names_out())
        if skip_vectorizer is not None
        else []
    )
    desc_names = (
        list(desc_vectorizer.get_feature_names_out())
        if desc_vectorizer is not None
        else []
    )
    return (
        [f"seq:{name}" for name in seq_names]
        + [f"skip:{name}" for name in skip_names]
        + [f"desc:{name}" for name in desc_names]
    )


def shorten_feature_name(name: str, max_len: int, tail_len: int) -> str:
    """Truncate a long feature name for display."""
    if max_len <= 0 or len(name) <= max_len:
        return name
    if tail_len <= 0 or tail_len >= max_len - 3:
        return f"{name[: max_len - 3]}..."
    head_len = max_len - tail_len - 3
    if head_len <= 0:
        return f"{name[: max_len - 3]}..."
    return f"{name[:head_len]}...{name[-tail_len:]}"
