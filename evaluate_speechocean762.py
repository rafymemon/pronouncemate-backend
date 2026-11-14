#!/usr/bin/env python3
"""
evaluate_speechocean762.py

Runs the PronounceMate pipeline (Pipeline.pipeline.run_pipeline) against
every utterance in the SpeechOcean762 corpus and computes:

  - Utterance-level PCC   : pipeline's overall session score vs the
                             human-rated utterance "accuracy" score
  - Word-level PCC        : pipeline's per-word score vs the human-rated
                             per-word "accuracy" score, pooled across
                             every word of every utterance
  - Basic error stats     : mean/MAE, failure list, timing

Dataset layout assumed (matches the folder listing you showed me:
WAVE/, train/, test/, resource/, README.md):

    speechocean762/
      WAVE/<speaker>/<utt_id>.WAV
      train/wav.scp, train/text, ...
      test/wav.scp,  test/text, ...
      resource/scores.json

====================================================================
 THINGS I GUESSED AND YOU MUST VERIFY BEFORE TRUSTING THE NUMBERS
====================================================================

1. SCORES.JSON SHAPE / SCALE (HUMAN_SCALE_MAX below)
   I'm going from memory of the public SpeechOcean762 release, NOT
   from your actual file (you haven't sent it). From memory, each
   utterance entry looks like:

       {
         "accuracy": 7, "completeness": 10, "fluency": 7,
         "prosodic": 7, "total": 7,
         "text": "FOR THE FIRST TIME",
         "words": [
           {"text": "FOR", "phones": ["F","AO1","R"],
            "phones-accuracy": [2,2,2], "mispronunciations": [],
            "accuracy": 10, "stress": 10, "total": 10},
           ...
         ]
       }

   with scores on a 0-10 scale. The script prints the FIRST entry it
   loads before running anything — read that printout and confirm the
   key names ("accuracy" vs something else) and the scale match. If
   they don't, fix HUMAN_SCALE_MAX and the two `entry.get("accuracy", ...)`
   lines accordingly. I'd rather you catch this from the printed sample
   than have me silently guess wrong and hand you a fabricated PCC.

2. wav.scp FORMAT
   I'm assuming Kaldi-style "<utt_id> <path>" lines. If your train/
   and test/ folders use a different file (e.g. only a `text` file,
   or IDs that don't match scores.json keys), load_wav_scp()/
   load_split_ids() will return nothing for that split and you'll see
   "0 utterances" — tell me what the file actually contains and I'll
   fix the parser.

3. DJANGO IMPORT PATH
   pipeline.py does `from .audio_processor import ...` etc., so it's a
   relative import inside a Django app package (your engineering notes
   call it the `Pipeline` app). This script therefore needs to run
   INSIDE your Django environment. I've added a django.setup() block
   below — you MUST set DJANGO_SETTINGS_MODULE (or edit the line) to
   your actual settings module, and the import line
   `from Pipeline.pipeline import run_pipeline` to match wherever
   pipeline.py actually lives if it's not literally that path.

Run a smoke test first:
    python evaluate_speechocean762.py --split test --limit 10

Then the real run:
    python evaluate_speechocean762.py --split test
    python evaluate_speechocean762.py --split all     # train+test combined
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
import gc
import torch

# ── Django bootstrap ─────────────────────────────────────────────
# EDIT THIS to your actual settings module (e.g. "backend.settings",
# "pronounce_mate.settings"). Without this, importing Pipeline.pipeline
# will fail on Django app registry errors.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django  # noqa: E402
django.setup()  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

# ── Dataset paths ──────────────────────────────────────────────
DATA_ROOT = Path("/home/developer/FYP-PronounceMate/Data/speechocean762")
WAVE_DIR = DATA_ROOT / "WAVE"
SCORES_PATH = DATA_ROOT / "resource" / "scores.json"

# ── Human score scale — VERIFY against the printed sample (see #1 above) ──
HUMAN_SCALE_MAX = 10.0


def load_wav_scp(split_dir: Path) -> dict:
    """Kaldi-style wav.scp: '<utt_id> <path-or-command>' per line -> {utt_id: path}."""
    mapping = {}
    wav_scp = split_dir / "wav.scp"
    if not wav_scp.exists():
        return mapping
    with open(wav_scp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, path = parts
            path = path.strip()
            candidates = [Path(path), DATA_ROOT / path, WAVE_DIR / path]
            resolved = next((c for c in candidates if c.exists()), None)
            mapping[utt_id] = str(resolved) if resolved else path
    return mapping


def load_split_ids(split_dir: Path) -> set:
    ids = set(load_wav_scp(split_dir).keys())
    if ids:
        return ids
    text_file = split_dir / "text"
    if text_file.exists():
        with open(text_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    ids.add(parts[0])
    return ids


def load_scores() -> dict:
    with open(SCORES_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_audio_path(utt_id: str, wav_maps: list) -> str:
    for m in wav_maps:
        if utt_id in m:
            return m[utt_id]
    for p in WAVE_DIR.rglob(f"{utt_id}.*"):
        return str(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the FIRST N utterances, sorted by id (deterministic smoke test — not a representative sample)")
    parser.add_argument("--sample", type=int, default=100,
                         help="randomly sample N utterances from the split")
    parser.add_argument("--seed", type=int, default=42,
                         help="random seed for --sample, so the sample is reproducible across runs (default: 42)")
    parser.add_argument("--out", default="speechocean762_results.json")
    args = parser.parse_args()
    if args.limit and args.sample:
        parser.error("--limit and --sample are mutually exclusive — pick one")

    # Adjust this import to wherever run_pipeline actually lives if
    # "Pipeline.pipeline" isn't the real module path.
    from Pipeline.pipeline import run_pipeline

    print("Loading scores.json ...")
    all_scores = load_scores()
    print(f"  {len(all_scores)} utterances with human scores")

    train_ids = load_split_ids(DATA_ROOT / "train")
    test_ids = load_split_ids(DATA_ROOT / "test")
    train_map = load_wav_scp(DATA_ROOT / "train")
    test_map = load_wav_scp(DATA_ROOT / "test")
    print(f"  train split: {len(train_ids)} ids   test split: {len(test_ids)} ids")

    if args.split == "train":
        utt_ids = train_ids
    elif args.split == "test":
        utt_ids = test_ids
    else:
        utt_ids = train_ids | test_ids

    utt_ids = sorted(utt_ids & set(all_scores.keys()))
    if not utt_ids:
        print("No utterance IDs matched between the split files and scores.json. "
              "Check #2 (wav.scp format) in the docstring above and tell me what "
              "train/wav.scp or train/text actually looks like.")
        sys.exit(1)

    if args.sample:
        rng = random.Random(args.seed)
        n = min(args.sample, len(utt_ids))
        full_n = len(utt_ids)
        utt_ids = sorted(rng.sample(utt_ids, n))  # sample from the FULL id list, then sort for stable logging order
        print(f"Randomly sampled {n} / {full_n} utterances (seed={args.seed}, reproducible)")
    elif args.limit:
        utt_ids = utt_ids[: args.limit]
    print(f"Evaluating {len(utt_ids)} utterances from split='{args.split}'")

    sample_id = utt_ids[0]
    print("\n── Sample ground-truth entry (VERIFY keys/scale before trusting results) ──")
    print(json.dumps(all_scores[sample_id], indent=2)[:1200])
    print("── end sample ──\n")

    utt_level = []   # (pipeline_score_0_100, human_score_0_100)
    word_level = []  # (pipeline_word_score_0_100, human_word_score_0_100)
    failures = []
    t0 = time.time()

    for i, utt_id in enumerate(utt_ids, 1):
        entry = all_scores[utt_id]
        reference_script = entry["text"]
        wav_path = resolve_audio_path(utt_id, [train_map, test_map])

        if not wav_path or not os.path.exists(wav_path):
            failures.append((utt_id, "audio file not found"))
            continue

        try:
            # skip_denoise=True: pipeline.py's own docstring says to set
            # this for benchmark corpora like SpeechOcean762 — see
            # audio_processor.preprocess_audio's docstring for why.
            result = run_pipeline(wav_path, reference_script, skip_denoise=True)
        except Exception as e:
            failures.append((utt_id, f"pipeline exception: {e!r}"))
            continue

        if not result.get("success", True):
            failures.append((utt_id, result.get("error", "pipeline returned failure")))
            continue

        human_utt_raw = entry.get("accuracy", entry.get("total"))
        if human_utt_raw is None:
            failures.append((utt_id, "no utterance-level 'accuracy'/'total' key in scores.json entry"))
            continue
        human_utt_score = human_utt_raw * (100.0 / HUMAN_SCALE_MAX)
        utt_level.append((result["score"], human_utt_score))

        # Word-level pairing: 'words' in the pipeline result follows
        # alignment order with 'inserted' entries interleaved; every
        # script word (incl. duplicates) gets exactly one non-inserted
        # entry, in order — see pipeline.py / error_detector.py. Strip
        # 'inserted' so what's left lines up 1:1 with scores.json's
        # word list, which is also in script order.
        pipeline_words = [w for w in result["words"] if w["type"] != "inserted"]
        gt_words = entry.get("words", [])

        if len(pipeline_words) != len(gt_words):
            failures.append((
                utt_id,
                f"word count mismatch: pipeline={len(pipeline_words)} gt={len(gt_words)} "
                f"(word-level pairing skipped for this utterance, utterance-level kept)"
            ))
        else:
            for pw, gw in zip(pipeline_words, gt_words):
                gt_word_raw = gw.get("accuracy", gw.get("total"))
                if gt_word_raw is None:
                    continue
                word_level.append((pw["per_word_score"], gt_word_raw * (100.0 / HUMAN_SCALE_MAX)))

        if i % 25 == 0 or i == len(utt_ids):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(utt_ids)}] elapsed={elapsed:.0f}s  "
                  f"ok={len(utt_level)} failed={len(failures)}")

        # Force PyTorch and Python to release memory back to the system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def pcc(pairs):
        if len(pairs) < 2:
            return None
        a, b = zip(*pairs)
        if len(set(a)) < 2 or len(set(b)) < 2:
            return None  # pearsonr is undefined for a constant series
        r, _ = pearsonr(a, b)
        return r

    utt_pcc = pcc(utt_level)
    word_pcc = pcc(word_level)


    # --- CALCULATE PRECISION, RECALL, F1 ---
    # Define what constitutes an "error" (e.g., score strictly below 80)
    ERROR_THRESHOLD = 80.0
    
    tp = 0  # True Positive: AI says error, Human says error
    fp = 0  # False Positive: AI says error, Human says correct
    fn = 0  # False Negative: AI says correct, Human says error
    tn = 0  # True Negative: AI says correct, Human says correct
    
    for ai_word_score, human_word_score in word_level:
        ai_is_error = ai_word_score < ERROR_THRESHOLD
        human_is_error = human_word_score < ERROR_THRESHOLD
        
        if ai_is_error and human_is_error:
            tp += 1
        elif ai_is_error and not human_is_error:
            fp += 1
        elif not ai_is_error and human_is_error:
            fn += 1
        elif not ai_is_error and not human_is_error:
            tn += 1
            
    # Prevent division by zero
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n================ RESULTS ================")
    print(f"Error Threshold      : < {ERROR_THRESHOLD}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1-Score             : {f1_score:.4f}")
    print(f"Confusion Matrix     : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Utterances evaluated : {len(utt_level)} / {len(utt_ids)}")
    print(f"Failures             : {len(failures)}")
    print(f"Utterance-level PCC  : {utt_pcc:.4f}" if utt_pcc is not None else "Utterance-level PCC  : n/a")
    print(f"Word-level PCC       : {word_pcc:.4f} (n={len(word_level)})" if word_pcc is not None else "Word-level PCC       : n/a")
    if utt_level:
        pipe_scores, human_scores = zip(*utt_level)
        print(f"Mean pipeline score  : {mean(pipe_scores):.1f}")
        print(f"Mean human score     : {mean(human_scores):.1f}")
        print(f"MAE (utterance)      : {mean(abs(p - h) for p, h in utt_level):.2f}")
    print("===========================================\n")

    if failures:
        print(f"First 10 failures (of {len(failures)}):")
        for utt_id, reason in failures[:10]:
            print(f"  {utt_id}: {reason}")
        print()

    out = {
        "split": args.split,
        "n_utterances": len(utt_ids),
        "n_evaluated": len(utt_level),
        "n_word_pairs": len(word_level),
        "n_failed": len(failures),
        "utterance_pcc": utt_pcc,
        "word_pcc": word_pcc,
        "mean_pipeline_score": mean([p for p, _ in utt_level]) if utt_level else None,
        "mean_human_score": mean([h for _, h in utt_level]) if utt_level else None,
        "failures": failures[:100],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()