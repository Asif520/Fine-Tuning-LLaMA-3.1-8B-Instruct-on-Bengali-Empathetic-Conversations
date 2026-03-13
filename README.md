# Fine-Tuning Llama 3.1-8B-Instruct on Bengali Empathetic Conversations  
[**Video Explanation**] ()

**Documentation & Technical Report**

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
- LoRA rank (`r`): 16 
- `lora_alpha`: 16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj 
- `use_gradient_checkpointing`: "unsloth" (saves ~30% VRAM)
- Quantization: 4-bit load + float16 compute (T4 does not support bfloat16 efficiently)
- Sequence length: 8192 (Full sequence)

## 3. Training Strategy & Hyperparameters

- **Dataset**: Bengali Empathetic Conversations Corpus (Kaggle) — ~38k QA pairs
- **Formatting**: Llama 3.1 Instruct chat template (`<|begin_of_text|>`, `<|eot_id|>`, headers)
- **Trainer**: `trl.SFTTrainer` + `SFTConfig`
- **Key hyperparameters** (final used):

```text
per_device_train_batch_size    = 2
gradient_accumulation_steps    = 8     → effective batch = 8
learning_rate                  = 5e-5  (reduced from 2e-4 to stabilize)
lr_scheduler_type              = "linear"
Max steps                      = 900
optim                          = "adamw_8bit"
weight_decay                   = 0.01
```
## 4. Challenges Faced & Solutions

- **Limited GPU Access**
  - Problem: More training steps and more expermints need more Gpu
  - Solution: first experment with less (100-500 steps), finialize the whole strategy and then train with 900 steps.

- **BLEU & ROUGE = 0.0**
  - Problem: Typical for open-ended empathetic responses (creative rephrasing)
  - Solution: Relied on perplexity drop + human evaluation (empathy score 1–5 on 10 samples)

- **Log history missing 'eval_loss'**
  - Problem: PEFT + SFTTrainer bug
  - Solution: Added trainer.can_return_loss = True + safe backward scan in logging

## 5. Results Summary & Analysis

- Final validation perplexity: ~2.1–2.4 (good drop from base model ~3.8+)
- BLEU / ROUGE-L: low (0.00–0.12 range) — expected on creative task
- Human evaluation (20 samples): average empathy score ~2.0-3.4/5

