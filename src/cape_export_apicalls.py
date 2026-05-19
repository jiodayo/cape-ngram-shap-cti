#!/usr/bin/env python3
"""Export CAPE reports into FFRI-like JSON files with apicalls and labels."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CAPE reports into apicalls JSON files for n-gram pipeline."
    )
    parser.add_argument(
        "--reports-zip",
        type=Path,
        default=Path("data/avast_ctu_cape/public_full_reports.zip"),
    )
    parser.add_argument(
        "--index-jsonl",
        type=Path,
        default=Path("data/avast_ctu_cape/cape_index.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/avast_ctu_cape/ngram_dataset"),
    )
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument(
        "--label-key", choices=["family", "type"], default="family"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Add address/path/registry meta tokens to each sample.",
    )
    parser.add_argument(
        "--year-split",
        type=int,
        default=None,
        help="Split by year: train <= YEAR, test > YEAR. Overrides train_ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-label-count", type=int, default=0)
    parser.add_argument("--api-max-len", type=int, default=2000)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--label-set-out", type=Path, default=None)
    parser.add_argument("--api-list-out", type=Path, default=None)
    parser.add_argument("--api-descriptions-out", type=Path, default=None)
    parser.add_argument(
        "--api-list-source", choices=["train", "all"], default="train"
    )
    return parser.parse_args()


def load_index(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def extract_api_sequence(report: Dict, max_len: int) -> List[str]:
    seq: List[str] = []
    behavior = report.get("behavior") or {}
    processes = behavior.get("processes") or []
    if not isinstance(processes, list):
        return seq
    for proc in processes:
        if not isinstance(proc, dict):
            continue
        calls = proc.get("calls") or []
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            api = call.get("api")
            if isinstance(api, str):
                seq.append(api)
                if max_len > 0 and len(seq) >= max_len:
                    return seq
    return seq


def extract_ip_tokens(ip_value: object) -> List[str]:
    if not isinstance(ip_value, str) or not ip_value:
        return []
    try:
        ip = ipaddress.ip_address(ip_value)
    except ValueError:
        return []

    tokens = []
    tokens.append("META_IP_V4" if ip.version == 4 else "META_IP_V6")
    if ip.is_loopback:
        tokens.append("META_IP_LOOPBACK")
    elif ip.is_private:
        tokens.append("META_IP_PRIVATE")
    elif ip.is_link_local:
        tokens.append("META_IP_LINKLOCAL")
    else:
        tokens.append("META_IP_PUBLIC")
    return tokens


def extract_port_tokens(port_value: object) -> List[str]:
    if not isinstance(port_value, int):
        return []
    if port_value <= 0 or port_value > 65535:
        return []

    tokens = []
    if port_value <= 1023:
        tokens.append("META_PORT_WELLKNOWN")
    elif port_value <= 49151:
        tokens.append("META_PORT_REGISTERED")
    else:
        tokens.append("META_PORT_DYNAMIC")

    common_ports = {22, 25, 53, 80, 110, 143, 443, 445, 3389, 8080}
    if port_value in common_ports:
        tokens.append(f"META_PORT_{port_value}")
    return tokens


def normalize_domain(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return ""
    value = value.split("/")[0]
    value = value.split(":")[0]
    return value.strip(".")


def extract_domain_tokens(domain_value: object) -> List[str]:
    if not isinstance(domain_value, str) or not domain_value:
        return []
    domain = normalize_domain(domain_value)
    if not domain:
        return []

    tokens = []
    if domain.count(".") >= 3:
        tokens.append("META_DOMAIN_DEEP")

    tld = ""
    if "." in domain:
        tld = domain.rsplit(".", 1)[-1]
    if tld:
        tld_map = {
            "com": "META_TLD_COM",
            "net": "META_TLD_NET",
            "org": "META_TLD_ORG",
            "info": "META_TLD_INFO",
            "ru": "META_TLD_RU",
            "cn": "META_TLD_CN",
            "jp": "META_TLD_JP",
            "kr": "META_TLD_KR",
            "de": "META_TLD_DE",
            "uk": "META_TLD_UK",
        }
        tokens.append(tld_map.get(tld, "META_TLD_OTHER"))
    return tokens


def extract_path_tokens(path_value: object) -> List[str]:
    if not isinstance(path_value, str) or not path_value:
        return []
    path = path_value.replace("/", "\\").lower()

    tokens = []
    if path.startswith("\\\\"):
        tokens.append("META_PATH_UNC")
    if path.startswith("c:\\") or path.startswith("d:\\"):
        tokens.append("META_PATH_DRIVE")
    if "\\windows\\system32" in path:
        tokens.append("META_PATH_SYSTEM32")
    elif "\\windows\\" in path:
        tokens.append("META_PATH_WINDOWS")
    if "\\program files" in path:
        tokens.append("META_PATH_PROGRAMFILES")
    if "\\users\\" in path:
        tokens.append("META_PATH_USERS")
    if "\\appdata\\" in path:
        tokens.append("META_PATH_APPDATA")
    if "\\temp\\" in path:
        tokens.append("META_PATH_TEMP")

    ext = os.path.splitext(path)[1]
    if ext:
        ext_map = {
            ".exe": "META_EXT_EXE",
            ".dll": "META_EXT_DLL",
            ".sys": "META_EXT_SYS",
            ".bat": "META_EXT_BAT",
            ".cmd": "META_EXT_CMD",
            ".ps1": "META_EXT_PS1",
            ".js": "META_EXT_JS",
            ".vbs": "META_EXT_VBS",
            ".doc": "META_EXT_DOC",
            ".docx": "META_EXT_DOC",
            ".xls": "META_EXT_XLS",
            ".xlsx": "META_EXT_XLS",
            ".pdf": "META_EXT_PDF",
            ".zip": "META_EXT_ARCHIVE",
            ".rar": "META_EXT_ARCHIVE",
            ".7z": "META_EXT_ARCHIVE",
            ".lnk": "META_EXT_LNK",
            ".tmp": "META_EXT_TMP",
        }
        tokens.append(ext_map.get(ext, "META_EXT_OTHER"))

    depth = path.count("\\")
    if depth <= 2:
        tokens.append("META_PATH_DEPTH_SHALLOW")
    elif depth <= 5:
        tokens.append("META_PATH_DEPTH_MED")
    else:
        tokens.append("META_PATH_DEPTH_DEEP")
    return tokens


def extract_registry_tokens(key_value: object) -> List[str]:
    if not isinstance(key_value, str) or not key_value:
        return []
    key = key_value.upper()
    tokens = []
    if key.startswith("HKEY_LOCAL_MACHINE") or key.startswith("HKLM"):
        tokens.append("META_REG_HKLM")
    elif key.startswith("HKEY_CURRENT_USER") or key.startswith("HKCU"):
        tokens.append("META_REG_HKCU")
    elif key.startswith("HKEY_CLASSES_ROOT") or key.startswith("HKCR"):
        tokens.append("META_REG_HKCR")
    elif key.startswith("HKEY_USERS") or key.startswith("HKU"):
        tokens.append("META_REG_HKU")
    elif key.startswith("HKEY_CURRENT_CONFIG") or key.startswith("HKCC"):
        tokens.append("META_REG_HKCC")
    else:
        tokens.append("META_REG_OTHER")

    depth = key.count("\\")
    if depth <= 1:
        tokens.append("META_REG_DEPTH_1")
    elif depth <= 3:
        tokens.append("META_REG_DEPTH_2_3")
    else:
        tokens.append("META_REG_DEPTH_4P")
    return tokens


def extract_meta_tokens(report: Dict) -> List[str]:
    tokens: List[str] = []
    network = report.get("network") or {}
    if isinstance(network, dict):
        for field in ("tcp", "udp"):
            entries = network.get(field)
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    tokens.extend(extract_ip_tokens(item.get("src")))
                    tokens.extend(extract_ip_tokens(item.get("dst")))
                    tokens.extend(extract_port_tokens(item.get("sport")))
                    tokens.extend(extract_port_tokens(item.get("dport")))

        domains = network.get("domains")
        if isinstance(domains, list):
            for item in domains:
                if isinstance(item, dict):
                    tokens.extend(extract_domain_tokens(item.get("domain")))
                elif isinstance(item, str):
                    tokens.extend(extract_domain_tokens(item))

        dns = network.get("dns")
        if isinstance(dns, list):
            for item in dns:
                if isinstance(item, dict):
                    tokens.extend(extract_domain_tokens(item.get("request")))

        http = network.get("http")
        if isinstance(http, list):
            for item in http:
                if not isinstance(item, dict):
                    continue
                tokens.extend(extract_domain_tokens(item.get("host")))
                tokens.extend(extract_port_tokens(item.get("port")))

    summary = report.get("behavior", {}).get("summary")
    if isinstance(summary, dict):
        file_fields = [
            "files",
            "read_files",
            "write_files",
            "delete_files",
            "executed_commands",
        ]
        for field in file_fields:
            values = summary.get(field)
            if isinstance(values, list):
                for item in values:
                    tokens.extend(extract_path_tokens(item))

        key_fields = ["keys", "read_keys", "write_keys", "delete_keys"]
        for field in key_fields:
            values = summary.get(field)
            if isinstance(values, list):
                for item in values:
                    tokens.extend(extract_registry_tokens(item))

    return sorted({token for token in tokens if token})


def split_records(
    records: List[Dict[str, str]],
    label_key: str,
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        label = rec.get(label_key)
        if isinstance(label, str) and label:
            by_label[label].append(rec)

    train: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []
    for _, items in by_label.items():
        rng.shuffle(items)
        if train_ratio >= 1.0:
            n_train = len(items)
        else:
            n_train = int(len(items) * train_ratio)
            n_train = max(1, n_train)
            if len(items) > 1:
                n_train = min(n_train, len(items) - 1)
        train.extend(items[:n_train])
        test.extend(items[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def parse_year(value: object) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    year_str = value[:4]
    if not year_str.isdigit():
        return None
    return int(year_str)


def split_records_by_year(
    records: List[Dict[str, str]],
    year_split: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    train: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []
    stats = {
        "missing_date": 0,
        "invalid_date": 0,
        "train_year_max": year_split,
        "test_year_min": year_split + 1,
    }

    for rec in records:
        year = parse_year(rec.get("date"))
        if year is None:
            date_val = rec.get("date")
            if date_val is None or date_val == "":
                stats["missing_date"] += 1
            else:
                stats["invalid_date"] += 1
            continue
        if year <= year_split:
            train.append(rec)
        else:
            test.append(rec)
    return train, test, stats


def export_split(
    records: List[Dict[str, str]],
    zf: zipfile.ZipFile,
    output_dir: Path,
    label_key: str,
    api_max_len: int,
    drop_empty: bool,
    include_meta: bool,
) -> Tuple[Counter, set, Dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    api_set: set = set()
    label_counts = Counter()
    stats = {
        "records": len(records),
        "written": 0,
        "missing_member": 0,
        "missing_label": 0,
        "read_errors": 0,
        "empty_apicalls": 0,
        "meta_tokens": 0,
    }

    for rec in records:
        member = rec.get("member")
        label = rec.get(label_key)
        if not isinstance(member, str) or not member:
            stats["missing_member"] += 1
            continue
        if not isinstance(label, str) or not label:
            stats["missing_label"] += 1
            continue

        sha = rec.get("sha256")
        if not isinstance(sha, str) or not sha:
            sha = Path(member).stem

        try:
            with zf.open(member) as f:
                report = json.load(f)
        except KeyError:
            stats["missing_member"] += 1
            continue
        except Exception:
            stats["read_errors"] += 1
            continue

        apicalls = extract_api_sequence(report, api_max_len)
        if not apicalls:
            stats["empty_apicalls"] += 1
            if drop_empty:
                continue

        meta_tokens: List[str] = []
        if include_meta:
            meta_tokens = extract_meta_tokens(report)
            stats["meta_tokens"] += len(meta_tokens)

        api_set.update(apicalls)
        if meta_tokens:
            api_set.update(meta_tokens)
        label_counts[label] += 1

        payload = {
            "sha256": sha,
            "family": rec.get("family"),
            "type": rec.get("type"),
            "functions": [label],
            "apicalls": apicalls,
            "source": "cape",
        }
        if include_meta:
            payload["meta_tokens"] = meta_tokens
        out_path = output_dir / f"{sha}.json"
        with out_path.open("w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=True)
        stats["written"] += 1

    return label_counts, api_set, stats


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()

    if not args.reports_zip.exists():
        raise FileNotFoundError(f"reports zip not found: {args.reports_zip}")
    if not args.index_jsonl.exists():
        raise FileNotFoundError(f"index JSONL not found: {args.index_jsonl}")

    output_root = args.output_root
    train_dir = args.train_dir or output_root / "train"
    test_dir = args.test_dir or output_root / "test"

    raw_records = load_index(args.index_jsonl)

    missing_label = 0
    label_counts = Counter()
    records: List[Dict[str, str]] = []
    for rec in raw_records:
        label = rec.get(args.label_key)
        if not isinstance(label, str) or not label:
            missing_label += 1
            continue
        label_counts[label] += 1
        records.append(rec)

    if args.min_label_count > 0:
        valid_labels = {
            label for label, count in label_counts.items() if count >= args.min_label_count
        }
        records = [rec for rec in records if rec.get(
            args.label_key) in valid_labels]
        label_counts = Counter(
            {label: count for label, count in label_counts.items()
             if label in valid_labels}
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    split_mode = "ratio"
    year_split_stats: Dict[str, int] | None = None
    if args.year_split is not None:
        split_mode = "year"
        train_records, test_records, year_split_stats = split_records_by_year(
            records, args.year_split
        )
    else:
        train_records, test_records = split_records(
            records, args.label_key, args.train_ratio, args.seed
        )

    with zipfile.ZipFile(args.reports_zip, "r") as zf:
        train_labels, train_api, train_stats = export_split(
            train_records,
            zf,
            train_dir,
            args.label_key,
            args.api_max_len,
            args.drop_empty,
            args.include_meta,
        )
        test_labels, test_api, test_stats = export_split(
            test_records,
            zf,
            test_dir,
            args.label_key,
            args.api_max_len,
            args.drop_empty,
            args.include_meta,
        )

    combined_labels = train_labels + test_labels
    api_set_all = set(train_api) | set(test_api)

    label_set_out = args.label_set_out or output_root / "label_set.json"
    api_list_out = args.api_list_out or output_root / "api.json"
    api_desc_out = args.api_descriptions_out or output_root / "api_descriptions.json"

    label_names = sorted(combined_labels)
    label_set = {name: idx for idx, name in enumerate(label_names)}

    api_set = train_api if args.api_list_source == "train" else api_set_all
    api_list = sorted(api_set)

    write_json(label_set_out, label_set)
    write_json(api_list_out, api_list)
    write_json(api_desc_out, {})

    summary = {
        "label_key": args.label_key,
        "split_mode": split_mode,
        "train_ratio": args.train_ratio,
        "include_meta": args.include_meta,
        "year_split": args.year_split,
        "year_split_stats": year_split_stats,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "min_label_count": args.min_label_count,
        "drop_empty": args.drop_empty,
        "records_in_index": len(raw_records),
        "records_with_label": len(records),
        "missing_label": missing_label,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "train_unique_labels": len(train_labels),
        "test_unique_labels": len(test_labels),
        "all_unique_labels": len(combined_labels),
        "api_vocab_train": len(train_api),
        "api_vocab_test": len(test_api),
        "api_vocab_all": len(api_set_all),
        "api_list_source": args.api_list_source,
        "outputs": {
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "label_set": str(label_set_out),
            "api_list": str(api_list_out),
            "api_descriptions": str(api_desc_out),
        },
    }

    write_json(output_root / "summary.json", summary)

    print("CAPE export complete")
    print(
        f"  train samples: {train_stats['written']} / {train_stats['records']}")
    print(
        f"  test samples:  {test_stats['written']} / {test_stats['records']}")
    print(f"  label_set: {label_set_out}")
    print(f"  api_list:  {api_list_out}")
    print(f"  summary:   {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
