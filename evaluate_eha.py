import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

from src.data_utils import (
    GOEMOTIONS_28,
    build_guided_attention_from_tokens,
    encode_texts,
    load_goemotions_csv,
    save_metadata,
)
from src.model import GraphLabelAttentionLayer, LabelDependencyLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EHA-BiLSTM-SEG model.")
    parser.add_argument("--model_dir", type=str, default="emotion-model-eha")
    parser.add_argument("--test_csv", type=str, default="data_test.csv")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_ablations", action="store_true")
    return parser.parse_args()


def load_artifacts(model_dir: Path):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    labels_payload = json.loads((model_dir / "label_classes.json").read_text(encoding="utf-8"))
    label_names = labels_payload["classes"] if isinstance(labels_payload, dict) else labels_payload

    model = tf.keras.models.load_model(
        model_dir / "emotion_model_eha.keras",
        custom_objects={
            "LabelDependencyLayer": LabelDependencyLayer,
            "GraphLabelAttentionLayer": GraphLabelAttentionLayer,
        },
        compile=False,
        safe_mode=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    return model, tokenizer, label_names, np.asarray(thresholds, dtype=np.float32), config


def predict_multilabel(
    model: tf.keras.Model,
    tokenizer,
    texts: List[str],
    max_len: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    enc = encode_texts(tokenizer, texts, max_len=max_len)
    guides = build_guided_attention_from_tokens(tokenizer, enc["input_ids"], enc["attention_mask"])
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "guide_attention": guides,
        }
    ).batch(batch_size)

    probs = []
    alphas = []
    for batch in ds:
        outputs = model(batch, training=False)
        probs.append(outputs["probs"].numpy())
        alphas.append(outputs["alpha_final"].numpy())
    return np.concatenate(probs, axis=0), np.concatenate(alphas, axis=0)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Dict:
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    micro_precision = float(
        precision_score(y_true, y_pred, average="micro", zero_division=0)
    )
    micro_recall = float(
        recall_score(y_true, y_pred, average="micro", zero_division=0)
    )
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    hamm = float(hamming_loss(y_true, y_pred))

    aurocs = []
    for c in range(y_true.shape[1]):
        y_c = y_true[:, c]
        if np.unique(y_c).shape[0] < 2:
            aurocs.append(None)
            continue
        aurocs.append(float(roc_auc_score(y_c, y_prob[:, c])))

    return {
        "subset_accuracy": subset_accuracy,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": hamm,
        "support": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "per_class_recall": per_class_recall,
        "auroc_per_class": aurocs,
    }


def perturb_misspell(text: str) -> str:
    words = text.split()
    if not words:
        return text
    out = []
    for w in words:
        if len(w) > 4:
            out.append(w[:-2] + w[-1] + w[-2])
        else:
            out.append(w)
    return " ".join(out)


def perturb_emoji_heavy(text: str) -> str:
    return f"{text} 😂 😭 😡 😱"


def perturb_sarcasm(text: str) -> str:
    return f"{text} yeah right /s"


def perturb_code_mixed(text: str) -> str:
    mapping = {"very": "bahut", "good": "bueno", "bad": "malo", "sad": "triste", "happy": "feliz"}
    words = text.split()
    return " ".join([mapping.get(w.lower(), w) for w in words])


def robustness_report(
    model,
    tokenizer,
    texts: List[str],
    y_true: np.ndarray,
    thresholds: np.ndarray,
    max_len: int,
    batch_size: int,
) -> Dict:
    base_probs, _ = predict_multilabel(model, tokenizer, texts, max_len=max_len, batch_size=batch_size)
    base_metrics = compute_metrics(y_true, base_probs, thresholds)
    base_macro = base_metrics["macro_f1"]

    perturbations = {
        "misspelled_text": perturb_misspell,
        "emoji_heavy_text": perturb_emoji_heavy,
        "sarcasm_heavy_subset": perturb_sarcasm,
        "code_mixed_sentences": perturb_code_mixed,
    }
    out = {}
    for name, fn in perturbations.items():
        noisy = [fn(t) for t in texts]
        probs, _ = predict_multilabel(model, tokenizer, noisy, max_len=max_len, batch_size=batch_size)
        metrics = compute_metrics(y_true, probs, thresholds)
        out[name] = {
            "macro_f1": metrics["macro_f1"],
            "macro_f1_drop": base_macro - metrics["macro_f1"],
        }
    return out


def attention_metrics(
    y_true: np.ndarray,
    alpha: np.ndarray,
    guide: np.ndarray,
) -> Dict:
    eps = 1e-9
    entropy = -np.sum(alpha * np.log(alpha + eps), axis=1).mean()
    topk = np.argsort(-alpha, axis=1)[:, :5]
    overlap = []
    for i in range(alpha.shape[0]):
        overlap.append(float(np.mean(guide[i, topk[i]] > 0)))
    lexicon_overlap = float(np.mean(overlap))
    human_alignment_proxy = float(np.mean([len(set(row.tolist())) / 5.0 for row in topk]))
    return {
        "attention_entropy": float(entropy),
        "emotion_lexicon_overlap": lexicon_overlap,
        "human_alignment_score_proxy": human_alignment_proxy,
    }


def run_ablation_protocol(
    model_dir: Path,
    test_csv: str,
    max_len: int,
    batch_size: int,
) -> Dict:
    from subprocess import run
    import sys
    import os

    train_csv = os.getenv("ABLATION_TRAIN_CSV", "data_train.csv")
    val_csv = os.getenv("ABLATION_VAL_CSV", "data_validation.csv")
    max_train_batches = os.getenv("ABLATION_MAX_TRAIN_BATCHES", "0")
    variants = [
        ("full", []),
        ("no_guided_attention", ["--no-use_guided_attention"]),
        ("no_label_dependency", ["--no-use_label_dependency"]),
        ("no_graph_label_attention", ["--no-use_graph_label_attention"]),
        ("no_multihead_attention", ["--no-use_multihead_attention"]),
        ("no_multitask", ["--no-use_multitask"]),
        ("no_contrastive", ["--no-use_contrastive"]),
        ("no_bilstm", ["--no-use_bilstm"]),
        ("no_transformer", ["--no-use_transformer"]),
    ]
    results = {}
    for name, extra_flags in variants:
        out = model_dir / f"ablation_{name}"
        cmd = [
            sys.executable,
            "train_eha.py",
            "--train_csv",
            train_csv,
            "--val_csv",
            val_csv,
            "--test_csv",
            test_csv,
            "--out_dir",
            str(out),
            "--max_len",
            str(max_len),
            "--batch_size",
            str(batch_size),
            "--epochs",
            "1",
            "--max_train_batches",
            str(max_train_batches),
            "--no-use_transformer",
        ] + extra_flags
        run(cmd, check=True)
        summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
        results[name] = summary
    return results


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    model, tokenizer, label_names, thresholds, config = load_artifacts(model_dir)
    dataset = load_goemotions_csv(Path(args.test_csv), label_names if label_names else GOEMOTIONS_28)

    probs, alpha = predict_multilabel(
        model=model,
        tokenizer=tokenizer,
        texts=dataset.texts,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    metrics = compute_metrics(dataset.labels, probs, thresholds)

    enc = encode_texts(tokenizer, dataset.texts, max_len=args.max_len)
    guide = build_guided_attention_from_tokens(tokenizer, enc["input_ids"], enc["attention_mask"])
    attn_report = attention_metrics(dataset.labels, alpha, guide)
    robust_report = robustness_report(
        model=model,
        tokenizer=tokenizer,
        texts=dataset.texts,
        y_true=dataset.labels,
        thresholds=thresholds,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    clf_report = classification_report(
        dataset.labels,
        (probs >= thresholds[None, :]).astype(np.int32),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    payload = {
        "main_metrics": metrics,
        "attention_metrics": attn_report,
        "robustness_metrics": robust_report,
        "classification_report": clf_report,
        "config": config,
    }

    if args.run_ablations:
        payload["ablation_results"] = run_ablation_protocol(
            model_dir=model_dir,
            test_csv=args.test_csv,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )

    out_path = model_dir / "evaluation_results.json"
    save_metadata(out_path, payload)
    print(f"Saved evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
