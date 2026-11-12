import os, json
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

CONFIG_PATH = "configs/efik_data_config.json"

def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_audio_dir"], exist_ok=True)

    ds = load_dataset(cfg["dataset_name"], split="train")
    ds = ds.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])

    def process(split, output_path, split_name):
        with open(output_path, "w", encoding="utf-8") as fh:
            for i, row in enumerate(tqdm(split)):
                audio = row["audio"]
                text = row["text"]
                if not text or audio is None:
                    continue
                
                sample_id = f"{cfg['language_id']}_{split_name}_{i:06d}"
                wav_path = os.path.join(cfg["output_audio_dir"], f"{sample_id}.wav")
                sf.write(wav_path, audio["array"], audio["sampling_rate"])
                fh.write(json.dumps({
                    "id": sample_id,
                    "audio_path": os.path.abspath(wav_path),
                    "text": text,
                    "language_id": cfg["language_id"],
                }, ensure_ascii=False) + "\n")

    process(ds["train"], cfg["output_train_manifest"], "train")
    process(ds["test"], cfg["output_dev_manifest"], "dev")
    print(f"Done. Train: {cfg['output_train_manifest']}, Dev: {cfg['output_dev_manifest']}")

if __name__ == "__main__":
    main()
