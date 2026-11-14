
import argparse
import json
import os
import random
import time

import jiwer
import pandas as pd
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

from data import SAMPLE_RATE, load_audio, load_split
from normalize import normalize_ka


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="optional LoRA adapter folder")
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=None, help="evaluate on a random subset (for quick checks)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--name", required=True, help="name used for the result files")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_model", default="openai/whisper-small",
                   help="original Whisper model to copy generation settings from, if the model's own are outdated "
                        "(use the matching size, e.g. openai/whisper-large-v2 for a large-v2 fine-tune)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(args.adapter or args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    # Older fine-tunes have an outdated generation config without language info; borrow it from the base model.
    if not getattr(model.generation_config, "lang_to_id", None):
        print(f"[eval] generation config is outdated, using the one from {args.base_model}")
        model.generation_config = GenerationConfig.from_pretrained(args.base_model)
    model.to(device).eval()

    items = load_split(args.cv_dir, args.split)
    if args.max_samples and args.max_samples < len(items):
        items = random.Random(args.seed).sample(items, args.max_samples)

    refs, preds, rows = [], [], []
    audio_seconds, start = 0.0, time.time()
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        audios = [load_audio(b["path"]) for b in batch]
        audio_seconds += sum(len(a) for a in audios) / SAMPLE_RATE
        feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.inference_mode():
            out = model.generate(feats.to(device, dtype), language="georgian", task="transcribe", max_new_tokens=225)
        texts = processor.batch_decode(out, skip_special_tokens=True)
        for b, t in zip(batch, texts):
            ref, pred = normalize_ka(b["text"]), normalize_ka(t)
            rows.append({"path": os.path.basename(b["path"]), "reference": b["text"], "prediction": t})
            if ref:
                refs.append(ref)
                preds.append(pred)
        print(f"\r{min(i + args.batch_size, len(items))}/{len(items)}", end="", flush=True)
    elapsed = time.time() - start
    print()

    metrics = {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "num_samples": len(refs),
        "wer": round(100 * jiwer.wer(refs, preds), 2),
        "cer": round(100 * jiwer.cer(refs, preds), 2),
        "real_time_factor": round(elapsed / audio_seconds, 4),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"{args.name}_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, f"{args.name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()