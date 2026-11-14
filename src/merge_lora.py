
import argparse

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.generation_config.language = "georgian"
    model.generation_config.task = "transcribe"
    model.save_pretrained(args.out)
    WhisperProcessor.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
