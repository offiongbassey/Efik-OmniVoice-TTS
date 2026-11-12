import os
import subprocess
import json

CONFIG_PATH = "configs/efik_tokenize_config.json"

def run_tokenizer(input_jsonl, tar_pattern, jsonl_pattern, cfg):
    cmd = [
        "python", "-m", "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl", input_jsonl,
        "--tar_output_pattern", tar_pattern,
        "--jsonl_output_pattern", jsonl_pattern,
        "--tokenizer_path", cfg["tokenizer_path"],
        "--nj_per_gpu", str(cfg["nj_per_gpu"]),
        "--shuffle", "True",
        "--loader_workers", "4",
        "--skip_errors",
        "--min_length", str(cfg["min_length"]),
        "--max_length", str(cfg["max_length"])
    ]

    print(f"{' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Tokenizer failed (exit {result.returncode})")
        print("\n".join(result.stderr.splitlines()[-20:]))
        return False
    return True

def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    train_tar_dir = os.path.join(cfg["output_token_dir"], "train")
    os.makedirs(train_tar_dir, exist_ok=True)
    ok_train = run_tokenizer(
        cfg["train_manifest"],
        os.path.join(train_tar_dir, "audios", "shard-%06d.tar"),
        os.path.join(train_tar_dir, "txts", "shard-%06d.jsonl"),
        cfg,
    )

    dev_tar_dir = os.path.join(cfg["output_token_dir"], "dev")
    os.makedirs(dev_tar_dir, exist_ok=True)
    ok_dev = run_tokenizer(
        cfg["dev_manifest"],
        os.path.join(dev_tar_dir, "audios", "shard-%06d.tar"),
        os.path.join(dev_tar_dir, "txts", "shard-%06d.jsonl"),
        cfg
    )

    if ok_train and ok_dev:
        print(f"Tokenization complete. Tokens are in {cfg['output_token_dir']}/ ")
    else:
        print("Tokenization failed.")


if __name__ == "__main__":
    main()
