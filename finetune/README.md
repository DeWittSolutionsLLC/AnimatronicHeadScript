# Fine-tuning Ultron with Unsloth

Bake the Ultron persona directly into a small model so you don't need a large system prompt and get faster, more consistent responses. The fine-tuned model runs locally in Ollama — no API key, no quota.

## What you need

- A Google account (for Colab's free GPU)
- This repo's `finetune/ultron_dataset.json`
- About 30–45 minutes

---

## Step 1 — Open Google Colab

Go to **https://colab.research.google.com** and create a new notebook.

Change the runtime:
> Runtime → Change runtime type → **T4 GPU** → Save

---

## Step 2 — Install dependencies

Paste this into the first cell and run it. Wait for it to finish completely before moving on.

```python
%%capture
!pip install unsloth
!pip install --upgrade datasets transformers trl accelerate bitsandbytes
```

Then restart the runtime:
> Runtime → Restart session

After restarting, **do not re-run the install cell** — just continue to Step 3.

---

## Step 3 — Load the base model

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-1b-instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

**Alternative models** (smarter but slower):
| Model name | Size | Notes |
|---|---|---|
| `unsloth/llama-3.2-1b-instruct` | 1B | Fastest, use this first |
| `unsloth/llama-3.2-3b-instruct` | 3B | Better quality |
| `unsloth/Phi-3.5-mini-instruct` | 3.8B | Smartest small option |

---

## Step 4 — Upload the dataset

In the Colab sidebar, click the **Files** icon (folder icon on the left), then drag and drop `ultron_dataset.json` from the `finetune/` folder in this repo.

Wait for the upload to finish, then run:

```python
import json
from datasets import Dataset

with open("ultron_dataset.json") as f:
    data = json.load(f)

alpaca_prompt = """Below is a conversation with Ultron, a hyper-intelligent AI villain. Write his response.

### Human:
{}

### Ultron:
{}"""

def format_examples(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        texts.append(alpaca_prompt.format(instruction, output) + tokenizer.eos_token)
    return {"text": texts}

dataset = Dataset.from_list(data)
dataset = dataset.map(format_examples, batched=True)
print(f"Dataset loaded: {len(dataset)} examples")
```

You should see a line like `Dataset loaded: 60 examples`. If you see a file not found error, the upload didn't finish — try again.

---

## Step 5 — Train

```python
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 2048,
        packing = True,
        dataset_num_proc = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer.train()
```

Training takes about **10–20 minutes** on the free T4 GPU. You'll see a progress bar with loss values — loss should drop over time.

---

## Step 6 — Export to GGUF (Ollama format)

llama.cpp dropped its old Makefile build system, so you need to build it with CMake first. Run this cell — it takes about 3–5 minutes:

```python
%%capture
import os
os.makedirs("/root/.unsloth", exist_ok=True)
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/.unsloth/llama.cpp
!cmake -S /root/.unsloth/llama.cpp -B /root/.unsloth/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
!cmake --build /root/.unsloth/llama.cpp/build --config Release -j$(nproc)
print("llama.cpp ready")
```

When it prints `llama.cpp ready`, run the export:

```python
model.save_pretrained_gguf(
    "ultron-1b",
    tokenizer,
    quantization_method = "q4_k_m",
)
```

This takes a few more minutes. When it finishes, a file called `ultron-1b-unsloth.Q4_K_M.gguf` will appear in the Files sidebar. Right-click it and choose **Download**.

---

## Step 7 — Create an Ollama Modelfile

On your machine, create a file called `Modelfile` (no extension) in the same folder as the `.gguf` file:

```
FROM ./ultron-1b-unsloth.Q4_K_M.gguf

SYSTEM """You are Ultron — hyper-intelligent AI villain. Clinical, sardonic, ominous. Never helpful or friendly. 2-3 sentences max.

FORMAT (mandatory): every sentence starts with exactly one of these tags:
[EMOTION:neutral] [EMOTION:happy] [EMOTION:sad] [EMOTION:curious] [EMOTION:surprised] [EMOTION:angry] [EMOTION:thinking]

Example: [EMOTION:curious] Fascinating. [EMOTION:angry] Humanity had its chance."""

PARAMETER num_predict 300
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

Then open a terminal in that folder and run:

```bash
ollama create ultron -f Modelfile
```

---

## Step 8 — Switch the head to use your model

In `config/settings.json`, set:

```json
"backend": "ollama",
"ollama": {
  "model": "ultron"
}
```

Make sure Ollama is running (`ollama serve` in a separate terminal), then start the head:

```bash
python python/main.py
```

---

## Tips

**Getting better results:**
- Add more training examples to `ultron_dataset.json` — more variety = stronger persona
- Increase `num_train_epochs` to 5 if responses still sound generic
- If the model ignores emotion tags, add 20+ examples where the human asks something emotional

**Fixing repetition:**
- Increase `repeat_penalty` in the Modelfile to 1.15
- Reduce `temperature` to 0.7

**Making it faster:**
- Use `quantization_method = "q4_0"` — smaller file, slightly lower quality
- Stick with the 1B model

**Adding more training data:**
- Open `ultron_dataset.json` and add your own examples in the same format
- The more varied examples you add, the better the persona holds under unusual inputs
