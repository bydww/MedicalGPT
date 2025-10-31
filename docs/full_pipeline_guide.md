# 从零开始运行 MedicalGPT 的 PT+SFT+DPO 全流程指南

> 目标：带你这个“什么都不懂的小白”一步步完成 **增量预训练(PT) → 有监督微调(SFT) → DPO 偏好对齐** 的全流程。
>
> 默认你已经把仓库代码放在 `/root/code/MedicalGPT`，并且这台机器上有至少两张可用 GPU（脚本示例使用 `CUDA_VISIBLE_DEVICES=0,1`）。如果你只有一张 GPU，稍后会告诉你如何改。

---

## 1. 准备工作

### 1.1 安装系统依赖

```bash
sudo apt update
sudo apt install -y git wget unzip python3 python3-venv python3-pip
```

- 如果你是在容器或已经准备好的训练机上，可以跳过已经安装的部分。

### 1.2 创建并激活 Python 虚拟环境

```bash
cd /root/code/MedicalGPT
python3 -m venv .venv
source .venv/bin/activate
```

- 每次重新登录后，都需要执行 `source .venv/bin/activate` 进入虚拟环境。

### 1.3 升级 pip 并安装项目依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- requirements.txt 中已经包含了 `transformers`、`accelerate`、`trl`、`peft` 等训练所需依赖。
- 如果出现安装失败，多半是因为网络或 CUDA 版本不兼容；可以重试或升级到合适的 CUDA+CUDNN 驱动。

### 1.4（可选）登录 Hugging Face Hub

- 如果你需要下载私有模型或数据，需要先登录：

  ```bash
  huggingface-cli login
  ```

  根据提示输入 Access Token。

---

## 2. 数据准备

### 2.1 PT 数据（增量预训练）

- PT 阶段读取 `./data/pretrain` 目录下的所有文件。
- 项目内置了示例：
  - `data/pretrain/en_article_tail500.txt`
  - `data/pretrain/fever.txt`
  - `data/pretrain/tianlongbabu.txt`
- 你可以往这个目录继续添加 `.txt` 或 `.jsonl`（格式为 `{"text": "..."}`）文件。
- 如果想验证数据格式，可以运行：

  ```bash
  python validate_jsonl.py --file data/pretrain/your_file.jsonl
  ```

### 2.2 SFT 数据（有监督微调）

- SFT 阶段读取 `./data/finetune` 目录。
- 示例：`data/finetune/medical_sft_1K_format.jsonl`，每行是一个 `{"conversations": [...]}` 对话对象。
- 如果你的原始数据是 Alpaca 格式，可以用脚本转换：

  ```bash
  python convert_dataset.py --in_file your_alpaca.json --out_file data/finetune/converted.jsonl
  ```

### 2.3 DPO 数据（直接偏好优化）

- DPO 阶段读取 `./data/reward` 目录。
- 示例：`data/reward/dpo_zh_500.jsonl`，包含 `question`、`response_chosen`、`response_rejected` 三个字段。
- 如果你有新的 DPO 数据，只要确保字段一致，放到该目录即可。

---

## 3. 了解关键脚本

| 阶段 | Python 脚本 | Shell 启动脚本 | 输出默认目录 |
| --- | --- | --- | --- |
| PT | `pretraining.py` | `run_pt.sh` | `outputs-pt-qwen-v1` |
| SFT | `supervised_finetuning.py` | `run_sft.sh` | `outputs-sft-qwen-v1` |
| DPO | `dpo_training.py` | `run_dpo.sh` | `outputs-dpo-qwen-v1` |
| RM (奖励模型) | `reward_modeling.py` | `run_rm.sh` | `outputs-rm-qwen-v1` |
| PPO (强化学习) | `ppo_training.py` | `run_ppo.sh` | `outputs-ppo-qwen-v1` |

### 3.1 统一训练策略（LoRA/QLoRA 等）

- **参数高效微调 (LoRA)**：所有阶段默认开启 `--use_peft True`，即使用 LoRA 方式在基座模型外层插入低秩适配器。示例配置中 LoRA 的秩为 8 (`--lora_rank 8`)，缩放因子 16 (`--lora_alpha 16`)，Dropout 0.05 (`--lora_dropout 0.05`)；这些值适合 7B 量级模型，可根据显存适当调节。
- **QLoRA 与 4bit 量化**：如果想进一步节省显存，可在对应脚本里的命令加 `--qlora True --load_in_4bit True --optim paged_adamw_32bit`。加上后，模型会以 4bit 量化加载，只微调 LoRA 层。
- **bf16 默认训练精度**：脚本里统一使用 `--torch_dtype bfloat16 --bf16`，在 A100、H100、4090 等 GPU 上表现稳定。如果你的硬件不支持 bfloat16，需要切换到 `fp16`，同时注意梯度溢出风险。
- **梯度累积**：所有脚本都通过 `--gradient_accumulation_steps` 控制有效批量。如果显存不足，先降低 `per_device_train_batch_size`，再增大累积步数，保持总 batch 大小。

### 3.2 训练流程依赖关系

1. `pretraining.py` 读取纯文本/JSONL 文件，执行增量预训练（可选但推荐）。
2. `supervised_finetuning.py` 在 PT 后模型上执行指令微调，核心利用 `template.py` 里的聊天模板将 ShareGPT 对话拼接成训练样本。
3. `reward_modeling.py` 以 SFT 模型为基座训练奖励模型，使用 `(chosen, rejected)` 对构造 pairwise loss；脚本里默认不使用 `torchrun`，直接单进程运行。
4. `ppo_training.py` 结合奖励模型与 policy 模型做强化学习；涉及 Trainer + TRL 的 PPO 算法，需同时加载 policy、ref policy、RM。
5. `dpo_training.py` 提供 ORPO 外的另一种偏好对齐方式，直接对 (chosen, rejected) 施加 DPO loss。
6. 每个阶段结束后都需要运行一次 `merge_peft_adapter.py` 把 LoRA 权重合并到基座模型里，供下一阶段继续训练。

> 每个阶段都会把 LoRA 权重保存在 `output_dir` 中。下一阶段开跑前必须用 `merge_peft_adapter.py` 把 LoRA 合并回基座模型。

---

## 4. 阶段一：增量预训练（PT）

### 4.1 根据实际 GPU 情况改脚本

打开 `run_pt.sh` 查看内容（可用 vim/nano 等编辑器）：

```bash
nano run_pt.sh
```

关键参数说明：
- `CUDA_VISIBLE_DEVICES=0,1`：使用第 0、1 号 GPU。如只有一张卡，请改为 `CUDA_VISIBLE_DEVICES=0`，并把 `--nproc_per_node 2` 改为 `1`。
- `--model_name_or_path`：预训练基座模型，示例使用 `Qwen/Qwen2.5-0.5B`。
- `--train_file_dir` / `--validation_file_dir`：增量预训练数据目录。
- `--max_train_samples`：示例限制为 10000 条用于快速跑通。正式训练时把它改成 `-1`，代表使用全部数据。

### 4.2 启动预训练

保存脚本后执行：

```bash
bash run_pt.sh 2>&1 | tee logs_pt.txt
```

- `tee` 会把日志同时输出到屏幕和 `logs_pt.txt`，方便检查。
- 训练完成后，权重保存在 `outputs-pt-qwen-v1` 目录。

### 4.3 合并 LoRA 权重

PT 阶段结束后，需要将 LoRA 合并回原模型以作为 SFT 的新基座：

```bash
python merge_peft_adapter.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --tokenizer_path Qwen/Qwen2.5-0.5B \
  --lora_model outputs-pt-qwen-v1 \
  --output_dir merged-pt-qwen-v1
```

- `--output_dir` 会生成一个可直接加载的新模型目录（包含合并后的权重和 tokenizer）。

---

## 5. 阶段二：有监督微调（SFT）

### 5.1 修改 `run_sft.sh`

打开脚本：

```bash
nano run_sft.sh
```

需要做的修改：
- 将 `--model_name_or_path` 改成上一步合并得到的目录，例如：
  ```
  --model_name_or_path ./merged-pt-qwen-v1
  ```
- 如果你使用的模型模板不是 `qwen`，记得修改 `--template_name` 对应的模板（具体可看 `template.py`）。
- 如果只有一张 GPU，同样需要改 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`。

### 5.2 启动 SFT

```bash
bash run_sft.sh 2>&1 | tee logs_sft.txt
```

- 训练完成后，LoRA 权重保存在 `outputs-sft-qwen-v1`。

### 5.3 合并 SFT 阶段 LoRA

```bash
python merge_peft_adapter.py \
  --base_model ./merged-pt-qwen-v1 \
  --tokenizer_path ./merged-pt-qwen-v1 \
  --lora_model outputs-sft-qwen-v1 \
  --output_dir merged-sft-qwen-v1
```

- 这个新目录将作为 DPO 阶段的 `--model_name_or_path`。

---

## 6. 阶段三：DPO 偏好对齐

### 6.1 修改 `run_dpo.sh`

```bash
nano run_dpo.sh
```

关键修改：
- `--model_name_or_path` 改成 `./merged-sft-qwen-v1`。
- `--template_name` 需匹配模型模板。例如 Qwen 使用 `qwen`。
- `--max_train_samples`、`--max_steps` 只是示例配置，正式训练可以调大或改成 `--num_train_epochs`。
- DPO 脚本默认用 `CUDA_VISIBLE_DEVICES=0,1`。如只有一张 GPU，把它改为 `0`，并删除或调整 `--gradient_accumulation_steps` 以满足显存。

### 6.2 启动 DPO

```bash
bash run_dpo.sh 2>&1 | tee logs_dpo.txt
```

- 输出目录 `outputs-dpo-qwen-v1` 中同样会生成 LoRA 权重。

### 6.3 合并 DPO 阶段 LoRA（得到最终模型）

```bash
python merge_peft_adapter.py \
  --base_model ./merged-sft-qwen-v1 \
  --tokenizer_path ./merged-sft-qwen-v1 \
  --lora_model outputs-dpo-qwen-v1 \
  --output_dir final-medgpt-dpo
```

- `final-medgpt-dpo` 就是可直接推理的最终模型目录。

---

## 7. 验证与推理

### 7.1 快速交互式推理

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --base_model ./final-medgpt-dpo \
  --template_name qwen \
  --interactive
```

- 如果你还有额外的 LoRA 想一起加载，可加 `--lora_model path_to_lora`（本例已经整合完，就不需要了）。

### 7.2 批量推理

- 把待测问题存成一个 `.jsonl` 文件（示例 `questions.jsonl`），每行 `{"prompt": "..."}`。
- 运行：

  ```bash
  CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model ./final-medgpt-dpo \
    --template_name qwen \
    --data_file questions.jsonl \
    --output_file answers.jsonl
  ```

---

## 8. 常见调整建议

- **只有单卡显存不够怎么办？**
  - 减小 `--per_device_train_batch_size`。
  - 增加 `--gradient_accumulation_steps`（相当于梯度累积）。
  - 开启 `--load_in_4bit True --qlora True`（需要同时设定 `--optim paged_adamw_32bit`）。在 `run_pt.sh` 等脚本中添加即可。
- **多机多卡训练**：需配置 `torchrun --nnodes` 等参数，详见 `docs/training_params.md` 中的“多机多卡训练”段落。
- **想扩充词表**：先用 `python build_domain_tokenizer.py` 训练新 tokenizer，再通过 `merge_tokenizers.py` 合并，PT 训练时记得加上 `--modules_to_save embed_tokens,lm_head`。
- **奖励模型与 PPO**：若要走完整 RLHF 流程，可先运行 `run_rm.sh` 训练奖励模型，再根据需要运行 `run_ppo.sh`。PPO 需要指定奖励模型路径，并确保 policy/ref policy 模型为上一步的 SFT 合并结果。
- **监控训练**：所有脚本默认把日志写入 `output_dir/logs/`，可以用 TensorBoard：

  ```bash
  tensorboard --logdir outputs-sft-qwen-v1/logs --host 0.0.0.0 --port 8008
  ```

---

## 9. 最终目录结构检查

完成之后，你应该在仓库里看到类似结构：

```
MedicalGPT/
├─ merged-pt-qwen-v1/        # PT 合并模型
├─ merged-sft-qwen-v1/       # SFT 合并模型
├─ final-medgpt-dpo/         # DPO 合并后的最终模型
├─ outputs-pt-qwen-v1/       # PT LoRA 权重
├─ outputs-sft-qwen-v1/      # SFT LoRA 权重
├─ outputs-dpo-qwen-v1/      # DPO LoRA 权重
└─ logs_*.txt                # 自己保存的训练日志
```

至此，你已经完整执行了一遍 **PT → SFT → DPO** 的训练流水线。遇到任何报错，优先查看命令行日志和对应的 `train_results.txt`、`trainer_state.json`。必要时可以把日志贴出来进一步排查。

祝训练顺利！
