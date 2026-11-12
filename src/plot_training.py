import re
import matplotlib.pyplot as plt
import os

LOG_PATH = "training_efik.log"
OUT_DIR = "docs/figures"

os.makedirs(OUT_DIR, exist_ok=True)

train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

with open(LOG_PATH, encoding="utf-8") as f:
    for line in f:
        # train loss is logged at the interval set by logging_steps in our config (20)
        m_train = re.search(r"Step (\d+) \| train/loss: ([\d.]+)", line)
        if m_train:
            train_steps.append(int(m_train.group(1)))
            train_losses.append(float(m_train.group(2)))

        m_eval = re.search(r"Running evaluation at step (\d+)", line)
        if m_eval:
            eval_steps.append(int(m_eval.group(1)))
        
        m_eval_loss = re.search(r"Eval Loss: ([\d.]+)", line)
        if m_eval_loss and eval_steps:
            # eval loss is logged on the line after the steps
            if len(eval_losses) < len(eval_steps):
                eval_losses.append(float(m_eval_loss.group(1)))

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Train loss", alpha=0.7)
if eval_steps and eval_losses:
    plt.plot(eval_steps[:len(eval_losses)], eval_losses, "o-", label="Eval loss", color="red")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Efik OmniVoice fine-tuning loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "training_loss_efik.png")
plt.savefig(out_path, dpi=200)
print(f"Figure saved at {out_path}")