**Project Focus**
- Domain-specific pipeline for training and aligning medical chat LLMs via Hugging Face Transformers plus PEFT/TRL utilities.
- Core stages live in `pretraining.py`, `supervised_finetuning.py`, `dpo_training.py`, `reward_modeling.py`, `ppo_training.py`, `orpo_training.py`, `grpo_training.py`.
- Shell wrappers `run_*.sh` encode sane defaults for GPUs, datasets, and LoRA hyperparameters.

**Workflow Overview**
- Execute stage scripts through the provided sh files; adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` before launch.
- Each stage expects LoRA adapters from the prior step to be merged into the base weights via `merge_peft_adapter.py` before reuse.
- Notebook pipelines `run_training_dpo_pipeline.ipynb` and `run_training_ppo_pipeline.ipynb` mirror the shell flow for quick experiments.

**Data & Formats**
- SFT data must follow the ShareGPT-style `{"conversations": [...]}` objects; see `data/finetune/medical_sft_1K_format.jsonl`.
- Reward/DPO datasets require `question`, `response_chosen`, `response_rejected` fields; sample in `data/reward/dpo_zh_500.jsonl`.
- PT corpora accept newline-delimited text or jsonl with `{"text": ...}` as shown in `data/pretrain/en_article_tail500.txt`.

**Templates & Prompting**
- `template.py` governs chat formatting; always set `--template_name` to match the base model family (e.g. `qwen`, `llama3`, `baichuan2`).
- Extending to new models means registering a `Conversation` via `register_conv_template` and reusing in the CLI.
- Template mismatches yield poor alignment; keep training and inference on the same template string.

**Training Conventions**
- Scripts rely on `Seq2SeqTrainingArguments`; `--bf16` is the default safe choice, drop to fp16 only when hardware demands it.
- LoRA defaults (`--target_modules all`, rank 8, alpha 16, dropout 0.05); override with architecture-specific lists (e.g. `q_proj,v_proj` for LLaMA/Mistral, `W_pack` for Baichuan).
- Enable QLoRA by pairing `--qlora True --load_in_4bit True` and switching the optimizer to `paged_adamw_32bit` for stability.

**Scaling & Memory**
- `check_and_optimize_memory()` auto-activates FlashAttention when `flash-attn` is installed; include `--flash_attn` to opt in.
- Large runs should lean on DeepSpeed configs (`zero2.json`, `zero3.json`); place them in repo root and reference via `--deepspeed`.
- Balance `per_device_*_batch_size` with `--gradient_accumulation_steps` to respect the VRAM table in `README.md`.

**Tokenizer & Vocab**
- Train domain tokenizers with `build_domain_tokenizer.py`, then merge into checkpoints via `merge_tokenizers.py` before PT.
- When expanding vocab during PT, add `--modules_to_save embed_tokens,lm_head` so later SFT runs remain compatible.
- Supporting resources live in `data/vocab/word_freq.txt` for frequency-driven merges.

**Inference & Serving**
- `inference.py` handles batch and interactive evaluation; merged weights go to `--base_model`, optional adapters via `--lora_model`.
- Web demos (`gradio_demo.py`, `fastapi_server_demo.py`) require the same template and tokenizer settings used in training.
- `chatpdf.py` layers RAG on top of fine-tuned models; drop source docs under `data/rag` before launching.

**Utilities & Validation**
- Normalize incoming instruction data with `convert_dataset.py`; it converts Alpaca-style JSON to the expected conversations schema.
- Run `validate_jsonl.py` on any new datasets to surface encoding or schema mismatches pre-flight.
- Logs and metrics land under each `output_dir` (TensorBoard traces in `logs/`, summaries in `train_results.txt`).

**Dependencies & Setup**
- Install base deps via `pip install -r requirements.txt`; RLHF pieces rely on `trl`, `accelerate`, and `peft` versions pinned there.
- Use `--cache_dir ./cache` across stages to reuse downloaded weights and tokenizers.
- Set `HF_HOME` or pass `--hf_hub_token` when training against private Hugging Face models.
