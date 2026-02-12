import csv
import io
import json
import os
import time
from collections import Counter
from datetime import datetime

from flask import Flask, redirect, render_template, request, send_file, url_for

try:
    from src.predict import predict_detailed
except ModuleNotFoundError:
    from predict import predict_detailed

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "emotion-dev-secret")

MAX_INPUT_CHARS = 1200
HISTORY_LIMIT = 30
CACHE_LIMIT = 500
TEXT_COL_CANDIDATES = ["text", "review", "content", "comment", "message"]

_PREDICTION_CACHE = {}
_CACHE_KEYS = []
_APP_HISTORY = []
_LAST_BATCH_RESULT = None


def _is_likely_english(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    ascii_letters = [ch for ch in letters if ord(ch) < 128]
    return (len(ascii_letters) / len(letters)) >= 0.7


def _validate_text(text):
    if not text or not text.strip():
        return "Please enter some text before analyzing."
    if len(text) > MAX_INPUT_CHARS:
        return f"Input is too long. Please keep it under {MAX_INPUT_CHARS} characters."
    return None


def _cache_get(key):
    return _PREDICTION_CACHE.get(key)


def _cache_set(key, value):
    if key in _PREDICTION_CACHE:
        _PREDICTION_CACHE[key] = value
        return
    _PREDICTION_CACHE[key] = value
    _CACHE_KEYS.append(key)
    if len(_CACHE_KEYS) > CACHE_LIMIT:
        evict = _CACHE_KEYS.pop(0)
        _PREDICTION_CACHE.pop(evict, None)


def _run_prediction(text, context, threshold=0.4):
    cache_key = (text.strip().lower(), context, threshold)
    cached = _cache_get(cache_key)
    if cached:
        result = dict(cached)
        result["from_cache"] = True
        return result

    start = time.perf_counter()
    details = predict_detailed(text, threshold=threshold, top_attention_tokens=10)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    details["all_probabilities_pct"] = [
        {
            "emotion": p["emotion"].upper(),
            "value": round(p["probability"] * 100, 2),
        }
        for p in details["all_probabilities"]
    ]
    details["top_predictions"] = details["all_probabilities_pct"][:3]
    details["processing_ms"] = elapsed_ms
    details["context"] = context
    details["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details["from_cache"] = False

    _cache_set(cache_key, details)
    return details


def _get_history():
    return _APP_HISTORY


def _set_history(history):
    global _APP_HISTORY
    _APP_HISTORY = history[-HISTORY_LIMIT:]


def _append_history(item):
    history = _get_history()
    history.append(item)
    _set_history(history)


def _history_distribution(history):
    counts = Counter([row["final_emotion"] for row in history if row.get("final_emotion")])
    return [{"emotion": k, "count": v} for k, v in counts.items()]


def _history_rows_for_export(history):
    rows = []
    for row in history:
        prob_map = {p["emotion"]: p["value"] for p in row.get("all_probabilities_pct", [])}
        rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "context": row.get("context", ""),
                "text": row.get("input_text", ""),
                "final_emotion": row.get("final_emotion", ""),
                "model_emotion": row.get("model_emotion", ""),
                "confidence_pct": row.get("confidence_pct", ""),
                "processing_ms": row.get("processing_ms", ""),
                "from_cache": row.get("from_cache", False),
                "probabilities": json.dumps(prob_map, ensure_ascii=False),
            }
        )
    return rows


def _batch_parse_texts(upload):
    if not upload or not upload.filename:
        return [], "Please choose a CSV or TXT file."

    filename = upload.filename.lower()
    raw = upload.read()
    if not raw:
        return [], "Uploaded file is empty."

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1")

    texts = []
    if filename.endswith(".txt"):
        texts = [line.strip() for line in decoded.splitlines() if line.strip()]
    elif filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(decoded))
        if reader.fieldnames is None:
            return [], "CSV file has no header row."

        lower_to_real = {h.strip().lower(): h for h in reader.fieldnames}
        chosen = None
        for candidate in TEXT_COL_CANDIDATES:
            if candidate in lower_to_real:
                chosen = lower_to_real[candidate]
                break
        if chosen is None:
            chosen = reader.fieldnames[0]

        for row in reader:
            value = str(row.get(chosen, "")).strip()
            if value:
                texts.append(value)
    else:
        return [], "Unsupported file type. Use .csv or .txt."

    if not texts:
        return [], "No valid text rows found in file."

    return texts, None


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error_message = None
    info_message = None
    language_warning = None
    text = ""
    context = "general"
    global _LAST_BATCH_RESULT
    batch_result = _LAST_BATCH_RESULT

    if request.method == "POST":
        form_mode = request.form.get("mode", "single")

        if form_mode == "batch":
            upload = request.files.get("batch_file")
            texts, parse_error = _batch_parse_texts(upload)
            if parse_error:
                error_message = parse_error
            else:
                rows = []
                dist = Counter()
                started = time.perf_counter()
                for idx, row_text in enumerate(texts, start=1):
                    validation_error = _validate_text(row_text)
                    if validation_error:
                        rows.append(
                            {
                                "row": idx,
                                "text": row_text,
                                "emotion": "SKIPPED",
                                "confidence": 0.0,
                                "error": validation_error,
                            }
                        )
                        continue

                    details = _run_prediction(row_text, context="batch", threshold=0.4)
                    emotion = details["final_emotion"].upper()
                    dist[emotion] += 1
                    rows.append(
                        {
                            "row": idx,
                            "text": row_text,
                            "emotion": emotion,
                            "confidence": round(details["confidence"] * 100, 2),
                            "processing_ms": details["processing_ms"],
                            "error": "",
                        }
                    )

                batch_result = {
                    "total_rows": len(rows),
                    "processed_rows": sum(1 for r in rows if not r["error"]),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "distribution": [{"emotion": k, "count": v} for k, v in dist.items()],
                    "rows": rows,
                }
                _LAST_BATCH_RESULT = batch_result
                info_message = f"Batch analysis completed for {len(rows)} rows."
        else:
            text = request.form.get("text", "").strip()
            context = request.form.get("context", "general").strip() or "general"
            threshold = 0.4

            error_message = _validate_text(text)
            if error_message is None:
                if not _is_likely_english(text):
                    language_warning = (
                        "Input seems non-English. Model is trained mostly on English text."
                    )

                details = _run_prediction(text, context=context, threshold=threshold)
                result = {
                    "input_text": details["input_text"],
                    "final_emotion": details["final_emotion"].upper(),
                    "model_emotion": details["model_emotion"].upper(),
                    "confidence_pct": round(details["confidence"] * 100, 2),
                    "processing_ms": details["processing_ms"],
                    "all_probabilities_pct": details["all_probabilities_pct"],
                    "top_predictions": details["top_predictions"],
                    "attention_tokens": details["attention_tokens"],
                    "timestamp": details["timestamp"],
                    "context": details["context"],
                    "from_cache": details["from_cache"],
                }
                _append_history(result)

    history = _get_history()
    history_distribution = _history_distribution(history)

    return render_template(
        "index.html",
        result=result,
        error_message=error_message,
        info_message=info_message,
        language_warning=language_warning,
        text=text,
        context=context,
        history=history,
        history_distribution=history_distribution,
        batch_result=batch_result,
        suggestion_min=20,
        suggestion_max=200,
        max_chars=MAX_INPUT_CHARS,
    )


def _export_rows_as_csv(rows, filename):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()) if rows else ["empty"])
    writer.writeheader()
    if rows:
        writer.writerows(rows)
    else:
        writer.writerow({"empty": "no_data"})

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )


def _export_rows_as_json(rows, filename):
    mem = io.BytesIO(json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype="application/json",
    )


@app.route("/export/history/<fmt>")
def export_history(fmt):
    rows = _history_rows_for_export(_get_history())
    if fmt == "csv":
        return _export_rows_as_csv(rows, "emotion_history.csv")
    if fmt == "json":
        return _export_rows_as_json(rows, "emotion_history.json")
    return redirect(url_for("home"))


@app.route("/export/batch/<fmt>")
def export_batch(fmt):
    batch = _LAST_BATCH_RESULT or {}
    rows = batch.get("rows", [])
    if fmt == "csv":
        return _export_rows_as_csv(rows, "emotion_batch_results.csv")
    if fmt == "json":
        return _export_rows_as_json(rows, "emotion_batch_results.json")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
