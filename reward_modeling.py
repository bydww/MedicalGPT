# -*- coding: utf-8 -*-  # 指定源码为UTF-8，避免中文注释乱码
"""
作者: XuMing (xuming624@qq.com)
用途: 奖励模型(Reward Model, RM)训练脚本

你将看到：
- 完整的参数解析与模型/分词器加载流程
- 数据集加载与成对样本(preferred vs rejected)构造
- DataCollator如何将成对输入打包并Padding到相同长度
- 自定义Trainer(RewardTrainer)如何计算InstructGPT的成对对比损失
- 所有关键函数与每行代码边注释，附带张量形状(A->B)与核心计算公式

核心损失: pairwise log-loss
    给定一对得分 r_c(优选) 与 r_r(弃选):
    Δ = r_c - r_r
    loss = - mean(log σ(Δ))，其中 σ(x) = 1 / (1 + e^{-x})
"""

import math  # 数学函数库，用于计算perplexity等
import os  # 操作系统接口，读写环境变量、路径
from dataclasses import dataclass, field  # dataclass用于定义参数容器，field指定默认值和帮助信息
from glob import glob  # 文件通配符匹配，递归搜集json/jsonl
from typing import Any, List, Union, Optional, Dict  # 类型提示，增强可读性与IDE支持

import torch  # PyTorch张量与深度学习框架
from torch.utils.data import Dataset  # Dataset基类类型提示用
from datasets import load_dataset  # Hugging Face datasets加载器
from loguru import logger  # 高级日志库
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training  # PEFT/LoRA工具
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评估指标:MSE/MAE
from transformers import (  # Transformers核心组件
    AutoConfig,  # 自动加载模型配置
    PreTrainedTokenizerBase,  # 分词器基类，用于类型标注
    AutoTokenizer,  # 自动加载分词器
    AutoModelForSequenceClassification,  # 序列分类模型，这里作为RM/VM输出标量分数
    HfArgumentParser,  # 参数解析器，将CLI映射到dataclass
    Trainer,  # 训练器基类，支持训练/评估/保存
    TrainingArguments,  # 训练参数容器
    set_seed,  # 设置随机种子
)
from transformers.trainer import TRAINING_ARGS_NAME

from template import get_conv_template  # 对话模板，将(history, question, answer)拼接为完整prompt字符串



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(  # 基座模型或检查点路径；RM从此加载分类头输出标量
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(  # 分词器路径；缺省则复用model_name_or_path
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "4bit量化加载以节省显存"})
    load_in_8bit: bool = field(default=False, metadata={"help": "8bit量化加载以节省显存"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,  # True使用Rust实现的高速分词器
        metadata={"help": "是否使用基于tokenizers库的Fast分词器"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",  # auto让transformers按显存自动切分到多GPU
        metadata={"help": "模型模块映射到设备的策略；auto自动分配"},
    )
    trust_remote_code: bool = field(
        default=True,  # 某些模型仓库自定义层实现需要打开
        metadata={"help": "加载远程仓库自定义代码时是否信任"},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(  # HF Hub数据集名称；None表示用本地json/jsonl
        default=None, metadata={"help": "datasets库可加载的数据集名称"}
    )
    dataset_config_name: Optional[str] = field(  # 数据集子配置，如语言/子任务
        default=None, metadata={"help": "datasets数据集配置名"}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练数据所在文件夹"})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "验证数据所在文件夹"}, )
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "prompt最大长度(含system/history/question/answer)"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "回答最大长度(用于上限)"})
    max_train_samples: Optional[int] = field(  # 限制训练样本数量用于调试或加速
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(  # 限制评估样本数量
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖datasets缓存"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,  # 若无validation split，则从train前1%划作验证
        metadata={"help": "当无验证集时，从训练集划分的百分比"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,  # map/filter的并行进程数
        metadata={"help": "预处理并行worker数量"},
    )


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "是否使用PEFT/LoRA微调"})
    target_modules: Optional[str] = field(default="all")  # 指定注入LoRA的模块名，逗号分隔；all表示自动发现
    lora_rank: Optional[int] = field(default=8)  # LoRA rank r
    lora_dropout: Optional[float] = field(default=0.05)  # LoRA dropout比例
    lora_alpha: Optional[float] = field(default=32.0)  # LoRA缩放alpha
    modules_to_save: Optional[str] = field(default=None)  # 额外需要保存的全量模块，如embed_tokens
    peft_path: Optional[str] = field(default=None)  # 已有LoRA权重路径，继续训练
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "prompt模板名称"})


def compute_metrics(eval_preds):
    """
    计算评估指标：MSE与MAE
    输入: eval_preds = (preds, labels)
        - preds: 预测值，形状(B, 1) 或 (B,)；来自RewardTrainer.prediction_step收集
        - labels: 真实标签（若提供），形状同preds
    输出: 字典 {"mse": MSE, "mae": MAE}

    公式：
        MSE = (1/B) * Σ_i (y_i - ŷ_i)^2
        MAE = (1/B) * Σ_i |y_i - ŷ_i|
    """
    preds, labels = eval_preds  # 解包元组
    # 某些情况下Trainer返回torch.Tensor，这里统一转为numpy便于sklearn计算
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()  # (B,1)张量 -> numpy数组
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()  # (B,1)张量 -> numpy数组
    mse = mean_squared_error(labels, preds)  # 计算均方误差MSE
    mae = mean_absolute_error(labels, preds)  # 计算平均绝对误差MAE

    return {"mse": mse, "mae": mae}  # 返回评估字典


@dataclass
class RewardDataCollatorWithPadding:
    """
    专用的成对数据整理器(DataCollator)，用于将(chosen, rejected)两路样本分开padding并打包成批次。
    输入features: List[Dict[str, Any]]，长度为B(批大小)
        每个feature包含：
            - input_ids_chosen: List[int]，长度L_c
            - attention_mask_chosen: List[int]，长度L_c
            - input_ids_rejected: List[int]，长度L_r
            - attention_mask_rejected: List[int]，长度L_r
    输出batch: Dict[str, Tensor]
        - input_ids_chosen: Tensor(B, L_max_c)
        - attention_mask_chosen: Tensor(B, L_max_c)
        - input_ids_rejected: Tensor(B, L_max_r)
        - attention_mask_rejected: Tensor(B, L_max_r)
        - return_loss: True  # 让Trainer走loss分支
    """
    tokenizer: PreTrainedTokenizerBase  # 用于执行pad
    padding: Union[bool, str] = True  # True或"max_length"等
    max_length: Optional[int] = None  # 若为int，则pad/截断到该长度
    pad_to_multiple_of: Optional[int] = None  # 对齐到该倍数，提高张量核利用
    return_tensors: str = "pt"  # 返回PyTorch张量

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []  # 收集优选样本字典列表
        features_rejected = []  # 收集弃选样本字典列表
        for feature in features:  # 遍历B条样本
            features_chosen.append({
                "input_ids": feature["input_ids_chosen"],  # List[int] (L_c)
                "attention_mask": feature["attention_mask_chosen"],  # List[int] (L_c)
            })
            features_rejected.append({
                "input_ids": feature["input_ids_rejected"],  # List[int] (L_r)
                "attention_mask": feature["attention_mask_rejected"],  # List[int] (L_r)
            })

        # 分别对两路进行padding，得到等长的(B, L_max_c)与(B, L_max_r)
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )  # 包含keys: input_ids(Tensor), attention_mask(Tensor)
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],  # (B, L_max_c)
            "attention_mask_chosen": batch_chosen["attention_mask"],  # (B, L_max_c)
            "input_ids_rejected": batch_rejected["input_ids"],  # (B, L_max_r)
            "attention_mask_rejected": batch_rejected["attention_mask"],  # (B, L_max_r)
            "return_loss": True,  # 触发Trainer.compute_loss
        }
        return batch  # 返回整理后的批数据


class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 前向得到两路的奖励分数：
        # inputs["input_ids_chosen"]: (B, Lc), inputs["attention_mask_chosen"]: (B, Lc)
        # 模型输出logits: (B, 1)  -> rewards_chosen: (B, 1)
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])[0]
        # inputs["input_ids_rejected"]: (B, Lr), inputs["attention_mask_rejected"]: (B, Lr)
        # 模型输出logits: (B, 1)  -> rewards_rejected: (B, 1)
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        # 计算损失：InstructGPT中的pairwise logloss
        # Δ = r_c - r_r,  σ(x) = 1 / (1 + e^{-x})
        # loss = - mean(log σ(Δ))  = mean(log(1 + e^{-Δ})) 的稳定实现
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 预测阶段：分别计算两路奖励分数，返回用于compute_metrics的张量
        device = model.device  # 当前模型所在设备

        inputs_chosen = {  # 优选路输入
            "input_ids": inputs["input_ids_chosen"].to(device),  # (B, Lc) -> device
            "attention_mask": inputs["attention_mask_chosen"].to(device),  # (B, Lc)
        }
        outputs_chosen = model(**inputs_chosen)  # logits: (B,1)
        rewards_chosen = outputs_chosen.logits.detach()  # (B,1) 取出张量并与计算图分离

        inputs_rejected = {  # 弃选路输入
            "input_ids": inputs["input_ids_rejected"].to(device),  # (B, Lr)
            "attention_mask": inputs["attention_mask_rejected"].to(device),  # (B, Lr)
        }
        outputs_rejected = model(**inputs_rejected)  # logits: (B,1)
        rewards_rejected = outputs_rejected.logits.detach()  # (B,1)

        # Keep the compute_loss method
        # 与训练相同的pairwise损失，便于eval_loss与perplexity计算
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """保存模型与分词器到输出目录。"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model  # 兼容DDP包装
    model_to_save.save_pretrained(output_dir)  # 保存权重与配置
    tokenizer.save_pretrained(output_dir)  # 保存分词器与特殊符号设置


class CastOutputToFloat(torch.nn.Sequential):
    """将模型输出强制转为float32，提升数值稳定性（未在当前流程中直接使用）。"""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


def print_trainable_parameters(model):
    """打印可训练参数数量与占比。"""
    trainable_params = 0  # 可训练参数总数
    all_param = 0  # 参数总数
    for _, param in model.named_parameters():  # 遍历所有参数张量
        all_param += param.numel()  # 累加元素个数
        if param.requires_grad:  # 仅统计需要梯度的部分
            trainable_params += param.numel()
    ratio = 100 * trainable_params / all_param if all_param > 0 else 0.0  # 占比%
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {ratio:.2f}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """自动发现可注入LoRA的线性层名称集合。
    - 当int4/int8为True时，匹配bitsandbytes的低比特线性层类型
    - 过滤输出头(lm_head)与评分(score)等不适合注入的位置
    返回: 排序后的模块名列表，如["q_proj", "v_proj", ...]
    """
    cls = torch.nn.Linear  # 默认匹配标准Linear
    if int4 or int8:
        import bitsandbytes as bnb  # 延迟导入，避免未安装时报错
        if int4:
            cls = bnb.nn.Linear4bit  # 4bit线性层
        elif int8:
            cls = bnb.nn.Linear8bitLt  # 8bit线性层
    lora_module_names = set()  # 去重集合
    for name, module in peft_model.named_modules():  # 遍历子模块
        if isinstance(module, cls):  # 定位线性层
            if 'lm_head' in name:  # 跳过最终输出头
                continue
            if 'score' in name:  # 某些评分头命名
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  # 取末级模块名
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments))  # 定义四类参数容器
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()  # 解析命令行到对象

    logger.info(f"Model args: {model_args}")  # 打印模型参数
    logger.info(f"Data args: {data_args}")  # 打印数据参数
    logger.info(f"Training args: {training_args}")  # 打印训练参数
    logger.info(f"Script args: {script_args}")  # 打印脚本附加参数
    logger.info(  # 记录分布式与精度信息
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    set_seed(training_args.seed)  # 固定随机种子以保证可复现

    # 加载奖励模型(序列分类，num_labels=1)
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )  # 指定加载dtype，"auto"交由权重决定
        world_size = int(os.environ.get("WORLD_SIZE", "1"))  # 分布式GPU数量
        if world_size > 1:  # 多卡时，将device_map映射到当前LOCAL_RANK
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        config = AutoConfig.from_pretrained(  # 载入配置并指定输出标签数为1
            model_args.model_name_or_path,
            num_labels=1,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(  # 加载权重到设备；可能为量化/分片
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(f"Error, model_name_or_path is None, RM must be loaded from a pre-trained model")

    # Load tokenizer
    tokenizer_kwargs = {  # 分词器加载参数
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path  # 优先用显式指定
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path  # 否则复用模型路径
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)  # 加载分词器
    prompt_template = get_conv_template(script_args.template_name)  # 获取prompt模板
    if tokenizer.eos_token_id is None:  # 确保EOS存在
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:  # 确保BOS
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:  # 确保PAD
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    if script_args.use_peft:  # 使用LoRA微调
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if script_args.peft_path is not None:  # 基于已有LoRA继续训练
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if model_args.load_in_8bit:  # 8bit场景下做k-bit训练准备（冻结某些层、设定dtypes）
                model = prepare_model_for_kbit_training(model)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:  # 自动发现线性层名
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,  # 序列分类任务
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)  # 将LoRA适配注入模型模块
        for param in filter(lambda p: p.requires_grad, model.parameters()):  # 确保可训练参数用fp32存储
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:  # 全参数微调
        logger.info("Fine-tuning method: Full parameters training")
        print_trainable_parameters(model)

    # Get reward dataset for tuning the reward model.
    if data_args.dataset_name is not None:  # 从Hub加载数据集
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}  # 本地文件模式
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    full_max_length = data_args.max_source_length + data_args.max_target_length  # 输入序列长度上限，用于过滤样本

    def preprocess_reward_function(examples):  # 将原始样本转换为(chosen, rejected)两路token序列
        """
        Turn the dataset into pairs of Question + Answer, where input_ids_chosen is the preferred question + answer
            and text_rejected is the other.
        """
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for system, history, question, chosen, rejected in zip(
                examples["system"],
                examples["history"],
                examples["question"],
                examples["response_chosen"],
                examples["response_rejected"]
        ):
            system_prompt = system or ""  # 无system则为空串
            chosen_messages = history + [[question, chosen]] if history else [[question, chosen]]  # [[q,a],...]
            chosen_prompt = prompt_template.get_prompt(messages=chosen_messages, system_prompt=system_prompt)  # str
            rejected_messages = history + [[question, rejected]] if history else [[question, rejected]]
            rejected_prompt = prompt_template.get_prompt(messages=rejected_messages, system_prompt=system_prompt)

            tokenized_chosen = tokenizer(chosen_prompt)  # 输出dict: input_ids(List[int]), attention_mask(List[int])
            tokenized_rejected = tokenizer(rejected_prompt)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])  # 累加到批(List[List[int]])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])  # 同上
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])  # 同上
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])  # 同上
        return new_examples

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:  # 训练集准备与token化
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):  # 仅主进程先执行
            tokenized_dataset = train_dataset.shuffle().map(  # 逐样本调用预处理 -> 生成四个字段
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            train_dataset = tokenized_dataset.filter(  # 过滤过长或空样本
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                    x['input_ids_chosen']) <= full_max_length
            )
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids_chosen']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:  # 验证集准备
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            tokenized_dataset = eval_dataset.map(
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = tokenized_dataset.filter(  # 同样过滤长度
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                    x['input_ids_chosen']) <= full_max_length
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids_chosen']))

    # Initialize our Trainer
    if training_args.gradient_checkpointing:  # 梯度检查点减少显存
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # 关闭KV cache以兼容检查点
    else:
        model.config.use_cache = True  # 保持推理友好
    model.enable_input_require_grads()  # 允许输入嵌入参与梯度（适配LoRA等）
    if torch.cuda.device_count() > 1:  # 多卡时禁用Trainer自带DP，改为module parallel
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=full_max_length, padding="max_length"  # 统一pad到定长，输出(B, full_max_length)
        ),
    )

    # Training
    if training_args.do_train:  # 进入训练
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")  # 展示一个批次形状
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  # 执行训练流程

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # 训练后恢复cache
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:  # 评估
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])  # PPL=exp(loss)，在RM上仅作参考
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
