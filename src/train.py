
import argparse
import os
import random
from dataclasses import dataclass

import jiwer
import torch
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data import SAMPLE_RATE, load_audio, load_split
from normalize import normalize_ka


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class Collator:

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, batch):
        audios = [load_audio(b["path"]) for b in batch]
        feats = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        tok = self.processor.tokenizer(
            [b["text"] for b in batch], padding=True, truncation=True, max_length=448, return_tensors="pt"
        )
        labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
        # The model adds the start token itself during training; drop it if the tokenizer added it.
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def make_compute_metrics(tokenizer):
    def compute(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        preds = [normalize_ka(t) for t in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [normalize_ka(t) for t in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        pairs = [(r, p) for r, p in zip(refs, preds) if r]
        refs, preds = [r for r, _ in pairs], [p for _, p in pairs]
        return {"wer": 100 * jiwer.wer(refs, preds), "cer": 100 * jiwer.cer(refs, preds)}
    return compute


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--output_dir", default="outputs/whisper-small-ka")
    p.add_argument("--mode", choices=["full", "lora"], default="full")
    p.add_argument("--load_in_8bit", action="store_true", help="LoRA only: load the base model in 8-bit")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--use_validated", action="store_true", help="train on validated.tsv minus dev/test speakers & sentences")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-5 (full) / 1e-3 (lora)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=500, help="dev clips used for evaluation during training")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in output_dir")
    args = p.parse_args()

    is_lora = args.mode == "lora"
    lr = args.lr or (1e-3 if is_lora else 1e-5)

    # ---- data ----
    train_items = load_split(args.cv_dir, "train", use_validated=args.use_validated)
    if args.max_train_samples:
        train_items = random.Random(args.seed).sample(train_items, min(args.max_train_samples, len(train_items)))
    dev_items = load_split(args.cv_dir, "dev")
    dev_items = random.Random(args.seed).sample(dev_items, min(args.eval_samples, len(dev_items)))

    # ---- model ----
    processor = WhisperProcessor.from_pretrained(args.model, language="georgian", task="transcribe")
    load_kwargs = {}
    if args.load_in_8bit:
        if not is_lora:
            raise SystemExit("--load_in_8bit only works with --mode lora")
        load_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"}
    model = WhisperForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.generation_config.language = "georgian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    decoder_start_token_id = model.config.decoder_start_token_id

    if is_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        # Needed so gradients flow through the frozen encoder input with gradient checkpointing.
        model.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ))
        model.print_trainable_parameters()

    # ---- training ----
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=min(500, args.max_steps // 10),
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=25,
        # Generating during eval is slow/fragile with 8-bit LoRA; use eval loss there and run evaluate.py after.
        predict_with_generate=not is_lora,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if is_lora else "wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,  # avoids re-spawning workers every epoch on Windows
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ListDataset(train_items),
        eval_dataset=ListDataset(dev_items),
        data_collator=Collator(processor, decoder_start_token_id),
        compute_metrics=None if is_lora else make_compute_metrics(processor.tokenizer),
    )
    processor.save_pretrained(args.output_dir)
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()