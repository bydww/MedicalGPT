# -*- coding: utf-8 -*-  # 指定UTF-8编码，避免中文注释乱码
"""
作者: XuMing (xuming624@qq.com)
用途: 基于已完成SFT的模型进行DPO(Direct Preference Optimization)训练

你将看到：
- 参数说明与解析，量化/LoRA/并行策略
- 分词器与模板处理、数据集加载、预处理映射
- 模型加载、梯度检查点、量化(QLoRA)配置
- DPO 训练配置与训练器初始化、训练/评估/保存
- 行内中文注释；在涉及计算时附公式与形状说明

核心目标（DPO损失）：给定同一提示x下的优选回答y+与弃选回答y-，定义
    Δ = β * [ (log π_θ(y+|x) - log π_θ(y−|x)) - (log π_ref(y+|x) - log π_ref(y−|x)) ]
    loss = - E_{(x,y+,y−)} [ log σ(Δ) ]，其中 σ(z) = 1 / (1 + e^{−z})
β(temperature) 控制KL偏好强度；π_ref通常为初始SFT策略或其冻结副本。
在TRL的DPOTrainer中会自动完成打分、差分与损失计算。
"""
import os  # 读取环境变量、判断路径
from copy import deepcopy  # 深拷贝模型作为参考策略（非PEFT时）
from dataclasses import dataclass, field  # dataclass用于参数容器
from glob import glob  # 递归匹配本地数据文件
from typing import Dict, Optional  # 类型提示

import torch  # 张量与模型训练
from datasets import load_dataset  # 加载Hub或本地数据集
from loguru import logger  # 日志库
from peft import LoraConfig, TaskType  # LoRA配置
from transformers import (  # Transformers核心组件
    AutoConfig,  # 模型配置加载
    AutoModelForCausalLM,  # 因果语言模型作为策略
    AutoTokenizer,  # 分词器
    HfArgumentParser,  # 参数解析器
    TrainingArguments,  # (未直接使用，保留导入)
    BitsAndBytesConfig,  # 4/8bit量化配置
)
from transformers.integrations import is_deepspeed_zero3_enabled  # 检测DeepSpeed ZeRO3
from trl import DPOTrainer, DPOConfig  # TRL中的DPO训练器与配置

from template import get_conv_template  # 对话模板，拼接system/history/question成prompt

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"  # 关闭并行以减少tokenizer警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许MKL重复加载，避免某些环境崩溃


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with DPO
    """
    # Model arguments
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "基础模型(SFT后)路径或HF名称，作为DPO策略π_θ"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "分词器路径；缺省则复用model_name_or_path"}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "8bit量化加载，显存更省"})
    load_in_4bit: bool = field(default=False, metadata={"help": "4bit量化加载，显存更省(配合QLoRA)"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "HF缓存目录，存放下载的模型/数据"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用Fast分词器(tokenizers实现)"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "覆盖默认dtype，auto表示按权重推断；可选bfloat16/float16/float32"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "模型模块映射；auto按显存自动切分到设备"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "加载远程自定义模型代码时是否信任"},
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "HF Hub数据集名称；None则读取本地json/jsonl"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集子配置，如语言/子任务"}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练数据文件夹(支持递归*.json/jsonl)"})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "验证数据文件夹"}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "对话模板名称，决定prompt格式"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "单卡训练批大小"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "单卡评估批大小"})
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "prompt最大长度上限(字符级粗控)"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "回答最大长度上限(字符级粗控)"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "回答最小长度(用于生成控制)"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "限制训练样本数用于调试或加速"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "限制评估样本数"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖datasets缓存"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "当无validation split时，从train划分的比例(%)"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "map/filter等预处理并行进程数"},
    )
    # Training arguments
    use_peft: bool = field(default=True, metadata={"help": "是否使用PEFT/LoRA"})
    qlora: bool = field(default=False, metadata={"help": "是否使用QLoRA(4bit+LoRA)"})
    target_modules: Optional[str] = field(default=None)  # 逗号分隔的LoRA注入目标名；含'all'表示自动发现
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)  # 已有LoRA权重路径
    do_train: bool = field(default=False, metadata={"help": "是否执行训练"})
    do_eval: bool = field(default=False, metadata={"help": "是否在验证集上评估"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "学习率"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "学习率调度器类型"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "warmup步数"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "权重衰减"})
    optim: Optional[str] = field(default="adamw_hf", metadata={"help": "优化器类型"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "是否使用fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "是否使用bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "是否启用梯度检查点(省显存)"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "梯度累积步数(等效大batch)"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "每隔多少步保存一次"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "每隔多少步评估一次"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "日志打印步间隔"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "输出目录"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "总训练步数(steps)"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "评估策略：steps/epoch/none"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "dataset到trainer过程中是否移除未使用列(保持False以保留prompt等)"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "日志后端：wandb或tensorboard"})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Parse args: {args}")

    # Load tokenizer
    # 说明：DPOTrainer会使用processing_class(=tokenizer)在内部完成tokenize与截断；
    # 这里先确保特殊符号完整（bos/eos/pad），避免后续拼接prompt+completion时丢terminator。
    # 形状预期（在trainer内部）：
    # - 对于每个样本i，存在三段文本：prompt_i、chosen_i、rejected_i。
    # - tokenizer会分别对拼接序列[tokenized(prompt_i + chosen_i)]与[tokenized(prompt_i + rejected_i)]编码。
    # - 记L_p为prompt长度，L_c/L_r为两个completion长度；批内padding到max_prompt_length/max_length。
    #   输入张量大致为：
    #   input_ids_chosen: [B, L_max]，attention_mask同形；logits: [B, L_max, V]。
    #   DPO仅对completion位置(去除prompt)的token计算对数似然差分。
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = get_conv_template(args.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # Get datasets
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    # 注意：此阶段仅在原始样本上构造三字段{prompt, chosen, rejected}（均为字符串），不做tokenize。
    # tokenize与截断由DPOTrainer根据DPOConfig在内部完成，减少重复存储。
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        """Load the paired dataset and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

                Prompts are structured as follows:
                    system_prompt + history[[q,a], [q,a]...] + question
                形状与数据流：
                - 输入字段：system: str | None, history: List[[q,a]], question: str。
                - 输出字段：
                    prompt: List[str]，每个元素为已格式化的对话前缀（长度约为字符级L_p_char）。
                    chosen/rejected: List[str]，对应两种回答（字符级长度L_c_char/L_r_char）。
                - DPOTrainer内部会将prompt与chosen/rejected逐一拼接后再tokenize，
                    在计算损失时仅对completion片段计算log pθ与log pref。
        """
        prompts = []
        for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
            system_prompt = system or ""
            history_with_question = history + [[question, '']] if history else [[question, '']]
            prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        # 这里的长度过滤使用字符长度近似控制（非token数），避免极端长样本；
        # 真正的token级截断在trainer侧按max_prompt_length/max_length执行。
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("First train example:")
        first_example = train_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        # 同训练集，字符级近似过滤
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("First eval example:")
        first_example = eval_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    # Load model
    # Load model
    # dtype选择：auto/None -> 由权重推断；否则按"bfloat16"/"float16"/"float32"映射到torch.*。
    # QLoRA场景：load_in_4bit=True时会构造BitsAndBytesConfig，权重以4bit量化+32bit优化器state进行训练。
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    logger.info(f"Device map: {args.device_map}")
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(f"Quantizing model, load_in_4bit: {args.load_in_4bit}, load_in_8bit: {args.load_in_8bit}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )
    # 将参与训练的参数显式转为FP32，规避部分模型在FP16下的数值/稳定性问题（如某些LayerNorm或LoRA权重初始化导致的溢出）。
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # Initialize our Trainer
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    # DPOConfig：控制token截断、batch、累计步数等；注意max_prompt_length+max_length应覆盖prompt+completion的token预算。
    training_args = DPOConfig(
        max_prompt_length=args.max_source_length,
        max_length=full_max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=args.remove_unused_columns,
        run_name=f"dpo_v1",
    )

    # Initialize DPO trainer
    peft_config = None
    if args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")
    # DPOTrainer数据流与损失：
    # - 对每个样本，计算Δ = β[(log πθ(y+|x) − log πθ(y−|x)) − (log πref(y+|x) − log πref(y−|x))]
    #   loss = − log σ(Δ)。其中log概率是沿completion token序列求和后得到的标量。
    # - ref_model：当未使用PEFT(full finetune)时，复制一份冻结的参考策略；使用PEFT时传None以节省显存，
    #   库侧会根据实现选择共享/复制base权重用于参考打分（具体行为依赖所用TRL版本）。
    trainer = DPOTrainer(
        model,
        ref_model=None if args.use_peft else deepcopy(model),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # 由trainer内部完成tokenize/截断/拼接
        peft_config=peft_config if args.use_peft else None,
    )
    print_trainable_parameters(trainer.model)

    # Training
    if args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            trainer.model.save_pretrained(args.output_dir)

    # Evaluation
    if args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
