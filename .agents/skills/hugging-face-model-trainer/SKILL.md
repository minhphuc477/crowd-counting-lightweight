---
name: hugging-face-model-trainer
description: >-
  Fine-tune and train vision & language models using TRL (Transformer Reinforcement Learning) and Unsloth on Hugging Face Jobs or local GPUs.
  Supports SFT, DPO, GRPO, Reward Modeling, QLoRA, and GGUF quantization.
---

# Hugging Face Model Trainer (TRL & Unsloth)

Comprehensive workflows for post-training, fine-tuning, and alignment.

## Training Paradigms

1. **SFT (Supervised Fine-Tuning)**: Standard instruction tuning on structured conversations/prompts (`SFTTrainer` / `SFTConfig`).
2. **DPO (Direct Preference Optimization)**: Preference alignment from paired chosen/rejected datasets (`DPOTrainer` / `DPOConfig`).
3. **GRPO (Group Relative Policy Optimization)**: Online rule-based / reasoning RL training for mathematical and algorithmic tasks.
4. **PEFT / LoRA / QLoRA**: Parameter-efficient fine-tuning with 4-bit / 8-bit quantization (`bitsandbytes`, `peft`).
5. **Unsloth Fast Fine-Tuning**: 2-5x faster execution with 60-80% VRAM reduction using fused Triton kernels (`FastLanguageModel`, `FastVisionModel`).

## Best Practices
- Ephemeral Cloud Runs: Always pass `push_to_hub=True` and `hub_model_id="org/repo"` with `secrets={"HF_TOKEN": "$HF_TOKEN"}`.
- Telemetry: Include Trackio or Weights & Biases for live loss and gradient tracking.
- Memory: Use gradient checkpointing, FlashAttention-2 / SDPA, and BF16/FP16 mixed precision.
