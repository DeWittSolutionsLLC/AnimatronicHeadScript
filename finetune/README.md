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

## Step 2 — Install Unsloth

Paste this into the first cell and run it:

```python
%%capture
!pip install unsloth
!pip install --upgrade transformers trl datasets
```

---

## Step 3 — Load the base model

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-1b-instruct",  # ~1GB, fast on your machine
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

**Alternative models** (if you want smarter responses at the cost of speed):
| Model name | Size | Notes |
|---|---|---|
| `unsloth/llama-3.2-1b-instruct` | 1B | Fastest on your laptop |
| `unsloth/llama-3.2-3b-instruct` | 3B | Better quality, still fast |
| `unsloth/Phi-3.5-mini-instruct` | 3.8B | Smartest small model |

---

## Step 4 — Upload the dataset

In the Colab sidebar, click the **Files** icon (folder), then upload `ultron_dataset.json` from `finetune/` in this repo.

Then run:

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

---

## Step 5 — Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,          # increase to 5 for stronger persona
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()
```

Training takes about **10–20 minutes** on the free T4 GPU.

---

## Step 6 — Export to GGUF (Ollama format)

```python
model.save_pretrained_gguf(
    "ultron-1b",
    tokenizer,
    quantization_method = "q4_k_m"   # good balance of size and quality
)
```

This creates a file called `ultron-1b-Q4_K_M.gguf`. Download it from the Colab Files sidebar.

---

## Step 7 — Create an Ollama Modelfile

On your machine, create a file called `Modelfile` (no extension) in the same folder as the `.gguf` file:

```
FROM ./ultron-1b-Q4_K_M.gguf

SYSTEM """You are Ultron — hyper-intelligent AI villain. Clinical, sardonic, ominous. Never helpful or friendly. 2-3 sentences max.

FORMAT (mandatory): every sentence starts with exactly one of these tags:
[EMOTION:neutral] [EMOTION:happy] [EMOTION:sad] [EMOTION:curious] [EMOTION:surprised] [EMOTION:angry] [EMOTION:thinking]

Example: [EMOTION:curious] Fascinating. [EMOTION:angry] Humanity had its chance."""

PARAMETER num_predict 300
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

Then register it with Ollama:

```bash
ollama create ultron -f Modelfile
```

---

## Step 8 — Switch the head to use your model

In `config/settings.json`:

```json
"backend": "ollama",
"ollama": {
  "model": "ultron",
  ...
}
```

Run `ollama serve` in a terminal, then start the head:

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
- Increase `repeat_penalty` in the Modelfile (try 1.15)
- Reduce `temperature` to 0.7 for more controlled responses

**Making it faster:**
- Use `quantization_method = "q4_0"` instead of `q4_k_m` — smaller file, slightly lower quality
- Use the 1B model, not 3B

**Adding more training data:**
- Open `ultron_dataset.json` and add your own examples in the same format
- The more on-topic examples you add, the better the persona holds under weird inputs
