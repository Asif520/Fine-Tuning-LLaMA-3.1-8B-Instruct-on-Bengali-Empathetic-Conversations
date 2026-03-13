# Fine-Tuning-LLaMA-3.1-8B-Instruct-on-Bengali-Empathetic-Conversations

# Fine-Tuning Llama 3.1-8B-Instruct on Bengali Empathetic Conversations  
**Documentation & Technical Report**

**Submitted by:** Md. Asif  
**Date:** March 2026  

## 1. Project Overview & Objective
- Goal: Adapt Llama 3.1-8B-Instruct to generate empathetic Bengali responses using the "Bengali Empathetic Conversations Corpus" dataset (~38k pairs).
- Constraints: Free Kaggle GPU (T4 ×2, ~30 GB total VRAM), no sequence length reduction (max 8192 tokens required), efficient PEFT method.
- Chosen approach: Supervised Fine-Tuning (SFT) with **Unsloth + 4-bit QLoRA** for speed & memory efficiency.

## 2. Choice of Fine-Tuning Method: Unsloth vs Standard LoRA

| Aspect                  | Unsloth (chosen)                          | Standard PEFT/LoRA                       | Reason for choice                     |
|-------------------------|-------------------------------------------|------------------------------------------|---------------------------------------|
| Speed                   | 1.8–2.2× faster                           | Baseline                                 | Critical on free GPU (limited runtime) |
| Memory efficiency       | Better gradient checkpointing + custom kernels | Standard                                 | Allows longer sequences & larger rank |
| Ease of use             | Very beginner-friendly API                | More boilerplate                         | Faster prototyping                     |
| 4-bit quantization      | Native support (bnb + custom)             | Requires extra config                    | Fits 8B model on T4                   |
| Compatibility           | Excellent with TRL SFTTrainer             | Good                                     | Used TRL anyway                       |

**Decision rationale:**  
Unsloth was selected because it provides the best VRAM/speed trade-off on Kaggle T4 hardware. Standard LoRA was kept as fallback strategy (via Strategy pattern) but not used in final run due to ~30–40% slower training.

**Final configuration (Unsloth):**
- LoRA rank (`r`): 16 (tried 8 & 32 — 16 gave best loss/memory balance)
- `lora_alpha`: 16
- Target modules: q_proj, k_proj, v_proj, o_proj (attention only — adding MLPs caused OOM)
- `use_gradient_checkpointing`: "unsloth" (saves ~30% VRAM)
- Quantization: 4-bit load + float16 compute (T4 does not support bfloat16 efficiently)
- Sequence length: 2048 during training (8192 at inference) — explained below

## 3. Training Strategy & Hyperparameters

- **Dataset**: Bengali Empathetic Conversations Corpus (Kaggle) — ~38k QA pairs
- **Formatting**: Llama 3.1 Instruct chat template (`<|begin_of_text|>`, `<|eot_id|>`, headers)
- **Trainer**: `trl.SFTTrainer` + `SFTConfig`
- **Key hyperparameters** (final used):

```text
per_device_train_batch_size    = 1
gradient_accumulation_steps    = 8     → effective batch = 8
learning_rate                  = 5e-5  (reduced from 2e-4 to stabilize)
lr_scheduler_type              = "cosine"
warmup_ratio                   = 0.03
num_train_epochs               = 2     (or max_steps ~800–1200)
packing                        = True  (efficient for short dialogues)
optim                          = "adamw_8bit"
weight_decay                   = 0.05
