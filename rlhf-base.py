import sys
sys.path.append(r"C:\Users\BMSCE CSE.DESKTOP-IUB6THA\Downloads\kshitij\UniEval")  # if needed to make sure your Python can import from the UniEval folder

from utils import convert_to_json
from metric.evaluator import get_evaluator

import transformers
import transformers.modeling_utils as _mod_utils

# if it's already there (unlikely), skip
if not hasattr(_mod_utils, "EncoderDecoderCache"):
    class EncoderDecoderCache:
        """
        Dummy placeholder so Seq2SeqTrainer can import it.
        No functional cache behavior — Trainer won’t actually use it.
        """
        def __init__(self, **kwargs): pass

    # inject into both the submodule and top‐level namespace
    _mod_utils.EncoderDecoderCache    = EncoderDecoderCache
    transformers.EncoderDecoderCache  = EncoderDecoderCache

# ════════════════════════════════════════════════════════════════
# Requirements:
#   pip install trl==0.7.4 transformers==4.38.2 peft==0.10.0 \
#               accelerate==0.28.0 bitsandbytes datasets evaluate pandas
# ════════════════════════════════════════════════════════════════
import os, gc, torch, pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────────────────────
# 1) UniEval multi‑dim evaluator (CPU only, load once)
# ────────────────────────────────────────────────────────────────
from utils import convert_to_json
from metric.evaluator import get_evaluator

sum_eval = get_evaluator("summarization", device="cpu")

@torch.inference_mode()
def unieval_4way(src, hyp, ref):
    """
    src, hyp, ref: lists of strings, length B
    returns: Tensor (B,4) with [coherence, consistency, fluency, relevance]
    """
    data = convert_to_json(
        output_list=hyp,
        src_list=src,
        ref_list=ref,
    )
    raw = sum_eval.evaluate(data, print_result=True)
    scores = [
        [d["coherence"], d["consistency"], d["fluency"], d["relevance"]]
        for d in raw
    ]
    return torch.tensor(scores, dtype=torch.float32)  # CPU (B,4)

# ────────────────────────────────────────────────────────────────
# 2) Load your SFT‑finetuned BART in 4‑bit + LoRA
# ────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFT_DIR = "/content/drive/MyDrive/bart_training/bart_clinical_ft"

# 2a) Quantize & prepare
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(
    SFT_DIR,
    quantization_config=bnb,
    device_map="auto",
)
base = prepare_model_for_kbit_training(base)
base.gradient_checkpointing_enable()
base.config.use_cache = False

# 2b) Attach fresh LoRA
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base, lora_cfg).to(DEVICE)

# 2c) Tokenizer (decoder‑only → left‑pad)
tok = AutoTokenizer.from_pretrained(SFT_DIR, use_fast=False)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model.config.pad_token_id = tok.eos_token_id
model.resize_token_embeddings(len(tok))

# 2d) Wrap for PPO
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model, peft_config=lora_cfg
).to(DEVICE)
ppo_ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model, peft_config=lora_cfg
).to(DEVICE).eval()
for p in ppo_ref_model.parameters():
    p.requires_grad = False

# ────────────────────────────────────────────────────────────────
# 3) Prepare your DataLoader (with references)
# ────────────────────────────────────────────────────────────────
df = pd.read_csv("/content/clinical_notes.csv")[["dialogue", "note"]]

class ClinDS(Dataset):
    def __init__(self, df, tok, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.L = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        conv = str(self.df.iloc[i]["dialogue"])
        ref = str(self.df.iloc[i]["note"])
        prompt = f"Summarize the following conversation:\n\n{conv}"
        enc = self.tok(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.L,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "src_txt": prompt,
            "ref_txt": ref,
        }

loader = DataLoader(
    ClinDS(df.sample(200, random_state=0), tok),
    batch_size=1, shuffle=True, pin_memory=True, drop_last=True
)

# ────────────────────────────────────────────────────────────────
# 4) Build PPOTrainer + optimizer
# ────────────────────────────────────────────────────────────────
ppo_cfg = PPOConfig(
    batch_size=1,
    mini_batch_size=1,
    log_with="tensorboard",
    project_kwargs={"logging_dir": "./logs"},
)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, ppo_model.parameters()),
    lr=2e-5
)

ppo_trainer = PPOTrainer(
    config=ppo_cfg,
    model=ppo_model,
    ref_model=ppo_ref_model,
    tokenizer=tok,
    optimizer=optimizer,
)

# ────────────────────────────────────────────────────────────────
# 5) Training loop with candidate generation and dominance rewards
# ────────────────────────────────────────────────────────────────
gen_kwargs = {
    "max_new_tokens": 64,
    "do_sample": True,
    "pad_token_id": tok.eos_token_id,
    "top_p": 0.9,
    "temperature": 0.7,
}

for epoch in range(1):
    for batch_idx, batch in enumerate(loader):
        # Prepare inputs
        ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        src_txt = batch["src_txt"]  # list[str]
        ref_txt = batch["ref_txt"]  # list[str]

        # Generate multiple candidates per prompt
        NUM_CANDIDATES = 1
        all_outs = []
        for _ in range(NUM_CANDIDATES):
            with torch.no_grad():
                out = ppo_model.generate(
                    input_ids=ids,
                    attention_mask=attn_mask,
                    **gen_kwargs
                )
            all_outs.append(out)

        # Stack outputs (B, K, L)
        outs = torch.stack(all_outs, dim=1)

        # Decode all candidates
        hyps = [
            [tok.decode(outs[b, k], skip_special_tokens=True)
            for k in range(NUM_CANDIDATES)]
            for b in range(outs.size(0))
        ]

        # Compute rewards using UniEval and dominance scoring
        rewards = []
        for b in range(len(src_txt)):
            # Get scores for all candidates (K, 4)
            scores = unieval_4way(
                [src_txt[b]] * NUM_CANDIDATES,
                hyps[b],
                [ref_txt[b]] * NUM_CANDIDATES
            ).numpy()

            # Compute dominance counts
            dom_counts = np.zeros(NUM_CANDIDATES)
            for i in range(NUM_CANDIDATES):
                for j in range(NUM_CANDIDATES):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(scores[i] >= scores[j]) and np.any(scores[i] > scores[j]):
                        dom_counts[i] += 1

            # Normalize to [-1, 1]
            max_dom = NUM_CANDIDATES - 1
            scalar_rewards = 2 * (dom_counts / max_dom) - 1
            rewards.append(scalar_rewards)

        # Flatten for PPO
        flat_queries = []
        flat_responses = []
        flat_rewards = []

        for b in range(len(src_txt)):
            for k in range(NUM_CANDIDATES):
                flat_queries.append(ids[b])
                flat_responses.append(outs[b, k])
                flat_rewards.append(torch.tensor([rewards[b][k]], device=DEVICE))

        # PPO step
        stats = ppo_trainer.step(
            queries=flat_queries,
            responses=flat_responses,
            scores=flat_rewards
        )

        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}")
            print(f"Sample output: {hyps[0][0][:100]}...")
            print(f"Average reward: {np.mean([r.item() for r in flat_rewards]):.4f}")

    print(f"✅ Epoch {epoch+1}/3 complete")

print("🎉 PPO fine-tuning done")
