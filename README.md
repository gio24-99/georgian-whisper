# Whisper for Georgian (ქართული) 🇬🇪

Fine-tuning OpenAI Whisper-small for Georgian speech recognition: **448% → 27.9% word error rate** on the Common Voice Georgian test set, about **40% fewer word errors than the previous public Georgian Whisper-small**, trained on a single laptop GPU.

<!-- DEMO VIDEO: on github.com, edit this file and drag your .mp4 here (under 10 MB) -->

**Model on Hugging Face:** [gio2499/whisper-small-ka](https://huggingface.co/gio2499/whisper-small-ka)

## Results

Mozilla Common Voice Scripted Speech 27.0, Georgian, official **test** split (14,258 clips).
Every model was evaluated with the same script (`src/evaluate.py`) and the same text normalization, so the numbers are directly comparable.

| Model | Params | WER ↓ | CER ↓ |
|---|---|---|---|
| openai/whisper-small (zero-shot) | 244M | 448.30 | 162.36 |
| sanchit-gandhi/whisper-small-ka-1k-steps (previous public model) | 244M | 46.33 | 12.22 |
| **gio2499/whisper-small-ka (this project)** | 244M | **27.92** | **8.81** |

WER = word error rate, CER = character error rate (lower is better). Raw metrics for each run are in [`results/`](results/).

### Key findings
- **Zero-shot Whisper-small is unusable for Georgian.** WER above 100% means it outputs more wrong words than the reference has; the model often gets stuck repeating phrases.
- **One fine-tuning run fixes this:** 27.9% WER and 8.8% CER, so roughly 9 out of 10 characters are correct.
- **Improvement over the previous public model** mostly comes from more data and longer training (5k steps on Common Voice 27.0 vs. 1k steps on Common Voice 11), not from a new method.
- **Dev and test results match** (28.7% vs. 27.9% WER), so there is no sign of overfitting.
- Fast enough for real-time use: the real-time factor is about 0.026 on an RTX 4070 Laptop GPU (around 38× faster than real time).

<!-- Add 5 example lines from results/whisper-small-ka_predictions.csv here: reference / before / after -->

## Approach

- **Data:** Common Voice Georgian from [Mozilla Data Collective](https://datacollective.mozillafoundation.org). Official train split (99,768 clips) for training; dev/test splits untouched. Common Voice splits are speaker-disjoint.
- **Text normalization** (`src/normalize.py`): Unicode NFC, Mtavruli → Mkhedruli, punctuation removed, whitespace collapsed. Applied identically to references and predictions.
- **Training** (`src/train.py`): Hugging Face Transformers `Seq2SeqTrainer`, full fine-tuning, 5,000 steps, effective batch size 16, learning rate 1e-5 with 500 warmup steps, fp16, gradient checkpointing, audio decoded on the fly. About 4.5 hours on one RTX 4070 Laptop GPU (8 GB).
- **Evaluation** (`src/evaluate.py`): greedy decoding with language forced to Georgian; WER/CER with `jiwer`.

## Reproduce

```bash
pip install -r requirements.txt   # install PyTorch with CUDA first

# 1. Download Common Voice Georgian from Mozilla Data Collective and extract it so that
#    data/cv-ka/ contains clips/, train.tsv, dev.tsv, test.tsv

# 2. Baseline
python src/evaluate.py --model openai/whisper-small --cv_dir data/cv-ka --name whisper-small-zeroshot

# 3. Train (add --resume to continue after an interruption)
python src/train.py --cv_dir data/cv-ka --output_dir outputs/whisper-small-ka --eval_samples 300

# 4. Evaluate and build the results table
python src/evaluate.py --model outputs/whisper-small-ka/final --cv_dir data/cv-ka --name whisper-small-ka
python src/make_table.py
```

**Demo:** `space/app.py` is a Gradio app (microphone or file upload). Run it locally with `python space/app.py` and open http://127.0.0.1:7860.

## Next experiments
- LoRA fine-tuning of `whisper-large-v3-turbo` on an 8-bit base (supported via `--mode lora --load_in_8bit`)
- Training on all validated clips minus dev/test speakers and sentences (`--use_validated`)
- Error analysis by sentence length, numbers and proper names

## Limitations

- Common Voice is read speech: short, clean, mostly Wikipedia sentences. Expect higher error rates on spontaneous speech, background noise, phone audio and regional dialects.
- Numbers, names and loanwords are common error sources.
- The full prediction files are not included because they contain dataset sentences; the Common Voice terms do not allow re-sharing the data.

## License

Code: MIT. Model weights follow the base Whisper license (MIT). Common Voice data is not redistributed here.
