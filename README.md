# Efik-OmniVoice-TTS

![Efik speaker using a phone](docs/figures/efik_speaker.png)
<p align="center">
  <img width="400" src="docs/figures/efik_speaker.png" alt="Efik speaker using a phone" />
</p>

Fine-tuning [OmniVoice](https://github.com/k2-fsa/OmniVoice) on 8.79 hours of Efik speech for low-resource tonal TTS.

**This repository is a fork of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice).** The original OmniVoice codebase is included here. My contributions are the Efik-specific data preparation script, tokenization configuration, fine-tuning configuration, and evaluation setup. The goal is to establish a baseline for codec-based TTS on an extremely low-resource tonal language, and to document where it fails.

The dataset used here is documented in the Interspeech 2026 paper: [Towards Digital Preservation of Efik: TTS for a Low-Resource African Language](https://arxiv.org/abs/2607.04515).

## Dataset

| Attribute | Value |
| :--- | :--- |
| **Source** | [offiongbassey/efik_speech_dataset](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset) |
| **Language** | Efik (`efi`) |
| **Size** | 5,875 clips, 8.79 hours |
| **Speakers** | 2 |
| **Split** | 90% train (5,287), 10% dev (588) |
| **Transcription** | Manual, reviewed by linguists and native speakers |

## Setup

```bash
conda create -n omnivoice-finetune python=3.10 -y
conda activate omnivoice-finetune

pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

```
Clone the repository and sync dependencies:
```bash

git clone https://github.com/offiongbassey/Efik-OmniVoice-TTS.git
cd Efik-OmniVoice-TTS
pip install -e .

pip install datasets huggingface_hub[cli] soundfile librosa pydub webdataset "torchcodec==0.7.*" matplotlib

```
Login to huggingface

```bash
hf auth login
```

## Data preparation
This script downloads the Efik dataset, splits it 90/10, saves the audio files locally, and writes the JSONL manifests that OmniVoice expects.

```bash
python src/prepare_efik_data.py
```
**It reads configs/efik_data_config.json and produces:**
- data/raw_audio/efi/ (5,875 .wav files)
- data/efik_train.jsonl (5,287 samples)
- data/efik_dev.jsonl (588 samples)

## Tokenization
This script converts the raw audio into discrete tokens using the OmniVoice tokenizer.
```bash
python src/tokenize_efik.py
```
**It reads configs/efik_tokenize_config.json and produces:**
- data/finetune/tokens/efi/train/ (audios and txts shards)
- data/finetune/tokens/efi/dev/ (audios and txts shards)

## Fine-tuning
**This is the command used to fine-tune OmniVoice on Efik.**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config configs/efik_train_config.json \
    --data_config examples/config/data_config_finetune.json \
    --output_dir exp/efik_omnivoice_finetune \
    > training_efik.log 2>&1 &
```

**Monitor with:**
```bash
tail -f training_efik.log
```

## Training details
| Parameter | Value |
| :--- | :--- |
| **Base model** | `k2-fsa/OmniVoice` |
| **Steps** | 5,000 |
| **Batch tokens** | 2,048 |
| **Gradient accumulation** | 4 |
| **Learning rate** | 1e-5 |
| **Mixed precision** | bf16 |
| **Hardware** | 1 GPU (22 GB) |
| **Training time** | 2h 12m |
| **Final eval loss** | 3.81 (down from 4.07) |


## Results
https://offiongbassey.github.io/efik-/pics/training_loss_efik.png

The loss curve shows the model learned quickly in the first 500 steps, then plateaued. Eval loss dropped from 4.07 to 3.81, indicating generalization to the dev set. The noise in the eval curve after step 2,000 suggests the model reached its capacity limit for this dataset size.

### The tonal finding 
The fine-tuned model produces intelligible Efik speech. However, it fails to preserve lexical tone.

A clear example is the word akpa. In the sentence "Mme mmọ emi ekenyenede udọñ ye ifet ndikot akpa edimiñ ñwed mbuk Mutanda Oyom Namondo", akpa means "first" (the first version of the book). But the model pronounces it as akpa meaning "cost" (as in "it cost me 500 naira"). The tonal contrast that distinguishes these meanings is lost.

This is the codec bottleneck. The discrete audio tokens produced by OmniVoice's neural codec discard the fine-grained pitch information required for lexical tone. 

## Inference
To generate a sample from the fine-tuned checkpoint:
```bash
omnivoice-infer \
    --model exp/efik_omnivoice_finetune/checkpoint-5000 \
    --text "Mme mmọ emi ekenyenede udọñ ye ifet ndikot akpa edimiñ ñwed mbuk Mutanda Oyom Namondo." \
    --output efik_test.wav
```

The generated audio is saved as efik_test.wav

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{edet2026efik,
  title={Towards Digital Preservation of Efik: TTS for a Low-Resource African Language},
  author={Edet, Offiong Bassey and Oyo-Ita, Emmanuel and Archibong, Archibong and Effanga, David and Awak, Mbuotidem},
  booktitle={Interspeech},
  year={2026},
  url={https://arxiv.org/abs/2607.04515}
}
```

## Acknowledgements
This project is built on top of OmniVoice, a state-of-the-art multilingual zero-shot TTS model developed by the k2-fsa team. The original codebase, model, and tokenizer are theirs.

If you use the base model, please also cite the OmniVoice paper:

```bibtex
@article{zhu2026omnivoice,
  title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
  author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
  journal={arXiv preprint arXiv:2604.00688},
  year={2026}
}
```

## Known limitations
This is a baseline. It is not the final model. The tonal errors are the point of the study.
- The corpus is small (8.79 hours, 2 speakers).
- The model does not preserve lexical tone.
- ASR models trained on this data may not generalize to noisy environments or unseen speakers.
- The model was fine-tuned on unmarked Efik text. Tone-marked text may improve results in future work.

