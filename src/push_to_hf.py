import os
import argparse
from huggingface_hub import HfApi

DEFAULT_CHECKPOINT = "exp/efik_omnivoice_finetune/checkpoint-5000"

CARD = """---
base_model: k2-fsa/OmniVoice
pipeline_tag: text-to-speech
language:
- efi
tags:
- text-to-speech
- tts
- low-resource
- african-languages
- efik
- tonal-language
license: other
---

# Efik-OmniVoice-TTS

Fine-tuned [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) on 8.79 hours of Efik speech.

This checkpoint is the baseline model on codec-based TTS for low-resource tonal languages. The training data used is documented in the Interspeech 2026 paper: [Towards Digital Preservation of Efik: TTS for a Low-Resource African Language](https://arxiv.org/abs/2607.04515).

## Training Details
- **Data:** 8.79 hours of Efik speech (5,875 clips, 2 speakers).
- **Steps:** 5,000
- **Loss:** Eval loss dropped from 4.07 to 3.81.

## Known Limitations
This is a baseline. It is not the final model. The tonal errors are the point of the study.

This model was fine-tuned on an extremely small corpus. While it produces intelligible Efik speech, it **fails to distinguish tonal minimal pairs**. For example, the word *akpa* can mean "first" or "cost" (as in "it cost me 500 naira"), depending on its tone. The model conflates these meanings. This is a known limitation of codec-based models for tonal languages, which is the focus of my ongoing research.

## Usage

    omnivoice-infer --model YOUR_REPO --text "Mma Effiom emesiere" --output out.wav

## Citation

    @inproceedings{edet2026efik,
      title={Towards Digital Preservation of Efik: TTS for a Low-Resource African Language},
      author={Edet, Offiong Bassey and Oyo-Ita, Emmanuel and Archibong, Archibong and Effanga, David and Awak, Mbuotidem},
      booktitle={Interspeech},
      year={2026},
      url={https://arxiv.org/abs/2607.04515}
    }
"""


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. offiongbassey/efik-omnivoice-tts")
    ap.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    try:
        print(f"Logged in as: {api.whoami()['name']}")
    except Exception as e:
        raise SystemExit(f"Not logged in to Hugging Face ({e}). Run: hf auth login")

    required = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.isfile(os.path.join(args.ckpt, f))]
    if missing:
        raise SystemExit(f"Checkpoint {args.ckpt} is missing required files: {missing}")

    plan = []
    for f in required:
        p = os.path.join(args.ckpt, f)
        plan.append((p, f))

    total = sum(os.path.getsize(p) for p, _ in plan)
    print(f"\nRepo: {args.repo} ({'PUBLIC' if args.public else 'private'})")
    print("Files to upload:")
    for p, r in plan:
        print(f"{human(os.path.getsize(p)):>10} {r}")
    print(f"{human(total):>10} TOTAL (+ README.md)")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled.")

    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)

    for p, r in plan:
        print(f"Uploading {r}...")
        api.upload_file(path_or_fileobj=p, path_in_repo=r, repo_id=args.repo, repo_type="model")

    card = CARD.replace("YOUR_REPO", args.repo)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo, repo_type="model")

    print(f"\nDone: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()

    