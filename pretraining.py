# -*- coding: utf-8 -*-  # 指定源文件编码，确保中文注释不会报错
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
# 版权声明，告知该文件归原作者及 HuggingFace 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 说明使用 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 使用者必须遵循许可证条款
# You may obtain a copy of the License at  # 指出许可证全文的获取方式
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache 许可证官方链接
#
# Unless required by applicable law or agreed to in writing, software  # 以下内容提醒该软件按“现状”提供
# distributed under the License is distributed on an "AS IS" BASIS,  # 软件无任何保证
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 使用者需查阅许可条款了解权利限制
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

part of this code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""  # 文件顶部的多行文档字符串，说明脚本用途与来源
import math  # 导入数学库，用于指数运算（如困惑度计算）
import os  # 导入操作系统库，用于环境变量和路径操作
from dataclasses import dataclass, field  # dataclass 用于快速定义参数容器类，field 用于设置默认值
from glob import glob  # glob 用于按模式匹配文件路径
from itertools import chain  # chain 用于将多个可迭代对象链接起来，常用于拼接 token 序列
from typing import Optional, List, Dict, Any, Mapping  # 类型注解，帮助理解函数参数类型

import numpy as np  # 数值计算库，用于处理数组
import torch  # PyTorch 深度学习框架，是核心模型训练库
from datasets import load_dataset  # HuggingFace datasets 库，用于加载数据集
from loguru import logger  # loguru 是一个高级日志库，用于输出调试信息
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training  # PEFT 相关组件，用于 LoRA 微调
from sklearn.metrics import accuracy_score  # sklearn 指标模块，用于计算准确率
from transformers import (  # transformers 库提供各种模型与训练工具
    AutoConfig,  # 自动加载模型配置
    AutoModelForCausalLM,  # 自动加载自回归语言模型
    AutoTokenizer,  # 自动加载分词器
    HfArgumentParser,  # HuggingFace 的命令行参数解析器
    Trainer,  # 通用训练循环类
    Seq2SeqTrainingArguments,  # Seq2Seq 训练参数类，这里也适用于 Causal LM
    is_torch_tpu_available,  # 判断 TPU 是否可用
    set_seed,  # 设置随机种子，保证复现
    BitsAndBytesConfig,  # bitsandbytes 量化配置，用于 4bit/8bit 加载
)
from transformers.trainer import TRAINING_ARGS_NAME  # 加载训练参数文件名常量，用于保存
from transformers.utils.versions import require_version  # 检查依赖库版本是否满足要求
from transformers.integrations import is_deepspeed_zero3_enabled  # 判断是否启用 DeepSpeed ZeRO-3


@dataclass  # 使用 dataclass 自动生成 __init__ 等方法，方便存储模型相关参数
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})  # 是否以 8bit 量化方式加载基座模型
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})  # 是否以 4bit 量化方式加载基座模型
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):  # dataclass 的钩子方法，在初始化后执行，用于做参数合法性检查
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass  # 定义与数据加载相关的参数集合
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):  # 检查是否启用 streaming，若启用则确认 datasets 版本满足要求
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass  # 定义脚本运行时用于控制 PEFT、LoRA、QLoRA 等配置的参数集合
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})  # 是否启用 PEFT（LoRA）微调
    target_modules: Optional[str] = field(default="all")  # 需要插入 LoRA 模块的层名称，all 代表自动推断
    lora_rank: Optional[int] = field(default=8)  # LoRA 低秩分解的秩 r
    lora_dropout: Optional[float] = field(default=0.05)  # LoRA 层的 dropout 概率
    lora_alpha: Optional[float] = field(default=32.0)  # LoRA 的缩放因子 α
    modules_to_save: Optional[str] = field(default=None)  # 额外需要保存的模块名称列表
    peft_path: Optional[str] = field(default=None)  # 若已有 LoRA 权重，可通过此路径加载继续训练
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})  # 是否启用 QLoRA（4bit 量化 + LoRA）


def accuracy(predictions, references, normalize=True, sample_weight=None):  # 计算分类准确率的辅助函数
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }  # accuracy_score 会比较预测与标签是否一致，normalize=True 表示返回比例


def compute_metrics(eval_preds):  # Trainer 在评估阶段会调用该函数计算额外指标
    preds, labels = eval_preds  # eval_preds 元组里包含预测张量和标签张量，形状均为 (batch_size, seq_len)
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics, we need to shift the labels
    labels = labels[:, 1:].reshape(-1)  # 将标签沿时间维右移一位并拉平，形状从 (B, T) -> (B, T-1) -> (B*(T-1))
    preds = preds[:, :-1].reshape(-1)  # 预测值去掉最后一个时间步，保持与标签对齐，并拉平
    return accuracy(predictions=preds, references=labels)  # 返回准确率字典


def preprocess_logits_for_metrics(logits, labels):  # 预处理模型输出的 logits，使之适配 compute_metrics
    if isinstance(logits, tuple):  # 某些模型会返回 (logits, past_key_values, ...)
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]  # 只取第一个元素作为真正的 logits，形状为 (batch_size, seq_len, vocab_size)
    return logits.argmax(dim=-1)  # 对最后一个维度取 argmax，得到预测的 token id，形状 (batch_size, seq_len)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:  # 自定义 data collator，确保容错性
    if not isinstance(features[0], Mapping):  # 如果样本是 dataclass 或对象而非字典
        features = [vars(f) for f in features]  # 将对象转换为字典
    first = features[0]  # 选取第一条样本参考其键结构
    batch = {}  # 初始化批次字典

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    if "label" in first and first["label"] is not None:  # 情况一：样本直接有 label 字段（标量）
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]  # 将张量转换成数值
        dtype = torch.long if isinstance(label, int) else torch.float  # 判断标签类型决定 dtype（分类 vs 回归）
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)  # 构造形状为 (batch_size,) 的标签张量
    elif "label_ids" in first and first["label_ids"] is not None:  # 情况二：样本具有 label_ids（序列标签）
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])  # [B, ...] 形状与原始张量一致
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float  # 根据元素类型决定 dtype
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)  # 创建标签张量

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():  # 遍历样本的每个键（例如 input_ids, attention_mask）
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):  # 排除标签和字符串（比如文本字段）
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])  # 将每条样本的张量堆叠 -> 形状 (B, ...)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))  # 将 numpy 数组堆叠并转为张量
                else:
                    batch[k] = torch.tensor([f[k] for f in features])  # 将标量或列表转为张量
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():  # 如果上面堆叠失败（可能因为长度不一致），退化为使用第一条样本重复
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))  # 重复第一条张量，形状 (B, ...)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))  # 重复 numpy 数组
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))  # 重复标量/列表

    return batch  # 返回整理好的批次字典，交由 Trainer 使用


class GroupTextsBuilder:  # 构造器，用于在 group_by_length 模式下拼接并切分文本
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length  # 保存模型最大序列长度（含首尾特殊 token）

    def __call__(self, examples):
        # Concatenate all texts.
        firsts = {k: examples[k][0][0] for k in examples.keys()}  # 记录每个字段的起始特殊 token（通常是 BOS），形状标量
        lasts = {k: examples[k][0][-1] for k in examples.keys()}  # 记录每个字段的结束特殊 token（通常是 EOS）
        contents = {k: sum([vi[1:-1] for vi in v], []) for k, v in examples.items()}  # 去除首尾特殊 token 将多个样本内容拼接
        total_length = len(contents[list(examples.keys())[0]])  # 拼接后内容的总长度（token 数）

        content_length = self.max_seq_length - 2  # 除去首尾 token 后每段内容的最大长度
        if total_length >= content_length:
            total_length = (total_length // content_length) * content_length  # 向下取整，确保能整段切分
        # Split by chunks of max_len.
        result = {
            k: [[firsts[k]] + t[i: i + content_length] + [lasts[k]] for i in range(0, total_length, content_length)] for
            k, t in contents.items()}  # 将内容按 content_length 切分，并重新补上首尾 token -> 每段长度为 max_seq_length
        return result  # 返回分段后的字典，每个值都是二维列表 [段数, max_seq_length]


class SavePeftModelTrainer(Trainer):  # 继承 HuggingFace Trainer，在保存模型时兼容 PEFT
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))  # 保存训练参数配置
        self.model.save_pretrained(output_dir)  # 调用模型的 save_pretrained（PEFT 模型会自动只保存 LoRA 参数）


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir  # 从训练参数中取输出目录
    os.makedirs(output_dir, exist_ok=True)  # 若路径不存在则创建

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model  # 如果模型被 DataParallel 包裹，则取其子模块
    model_to_save.save_pretrained(output_dir)  # 保存完整模型权重
    tokenizer.save_pretrained(output_dir)  # 保存分词器，包含词表等信息


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir  # 输出目录
    os.makedirs(output_dir, exist_ok=True)  # 创建目录
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()  # DeepSpeed ZeRO-3 需要特定方式聚合权重
    model_to_save = model.module if hasattr(model, "module") else model  # 取实际模型
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)  # 保存合并后的权重
    tokenizer.save_pretrained(output_dir)  # 保存分词器


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0  # 可训练参数计数器
    all_param = 0  # 总参数计数器
    for _, param in model.named_parameters():  # 遍历模型的全部参数
        all_param += param.numel()  # numel() 返回参数张量的元素总数
        if param.requires_grad:
            trainable_params += param.numel()  # 仅累加 requires_grad=True 的参数
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )  # 输出统计信息，帮助确认 LoRA 是否生效


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear  # 默认查找 nn.Linear 层
    if int4 or int8:
        import bitsandbytes as bnb  # 引入 bitsandbytes 中的量化线性层定义
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():  # 遍历模型模块层级
        if isinstance(module, cls):  # 找到匹配的线性层
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue  # 跳过输出层，避免修改最终预测头
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  # 记录层名称（末级名称）
    return sorted(lora_module_names)  # 返回排序后的列表，便于日志查看


def main():  # 脚本入口函数
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))  # 定义命令行参数解析，其中包含四类 dataclass
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()  # 将命令行参数解析为对应 dataclass 实例

    # Remove the explicit distributed initialization and simplify the process check
    # The Trainer will handle distributed training setup
    is_main_process = training_args.local_rank in [-1, 0]  # 判断当前进程是否为主进程（单卡或 rank 0）

    # Only log on main process
    if is_main_process:
        logger.info(f"Model args: {model_args}")  # 输出模型相关参数
        logger.info(f"Data args: {data_args}")  # 输出数据相关参数
        logger.info(f"Training args: {training_args}")  # 输出训练超参数
        logger.info(f"Script args: {script_args}")  # 输出自定义脚本参数
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )  # 输出分布式信息（local_rank、GPU 数量、是否使用 fp16）

    # Set seed before initializing model.
    set_seed(training_args.seed)  # 设置随机种子，确保多次运行结果一致

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,  # 指定缓存目录，避免重复下载
        "use_fast": model_args.use_fast_tokenizer,  # 是否使用 fast tokenizer（依赖 Rust）
        "trust_remote_code": model_args.trust_remote_code,  # 是否信任远程仓库自定义代码
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path  # 如果未指定单独的 tokenizer，则复用模型路径
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)  # 加载分词器配置与词表

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length  # 如果未指定 block_size，则默认使用模型允许的最大长度
        if block_size > 2048:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )  # 提示用户显式指定 block_size，以免显存不足
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )  # 如果用户给的 block_size 超过模型上限，发出警告
        block_size = min(data_args.block_size, tokenizer.model_max_length)  # 实际使用的 block_size 不超过模型上限

    # Preprocessing the datasets.
    def tokenize_function(examples):  # 针对普通（非 streaming）模式，带 padding 的分词函数
        tokenized_inputs = tokenizer(
            examples["text"],  # 输入文本列表，长度为 batch_size，形状 [batch_size]
            truncation=True,  # 超出 block_size 会自动截断
            padding='max_length',  # 对齐到固定长度 block_size，输出张量形状 (batch_size, block_size)
            max_length=block_size
        )
        # Copy the input_ids to the labels for language modeling. This is suitable for both
        # masked language modeling (like BERT) or causal language modeling (like GPT).
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # 语言模型训练中标签等于输入，形状相同

        return tokenized_inputs  # 返回包含 input_ids、attention_mask、labels 的字典

    def tokenize_wo_pad_function(examples):  # 针对 group_by_length 模式的分词函数，不做 padding
        return tokenizer(examples["text"])  # 返回长度不一的 token 序列列表

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_text_function(examples):  # 将若干段文本拼接后按 block_size 切块
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 将多个列表拼接成单一长列表
        total_length = len(concatenated_examples[list(examples.keys())[0]])  # 获取拼接后的总长度
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size  # 只保留能整除 block_size 的部分，避免最后一段过短
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }  # 切分后的每个列表形状 [num_blocks, block_size]
        result["labels"] = result["input_ids"].copy()  # 标签与输入相同
        return result

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}  # 存放训练/验证文件路径
        dataset_args = {}  # 存放额外的加载参数
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.txt', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")  # 打印找到的训练文件列表
            # Train data files must be same type, e.g. all txt or all jsonl
            types = [f.split('.')[-1] for f in train_data_files]
            if len(set(types)) > 1:
                raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")  # 确保格式一致
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.txt', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
            # Train data files must be same type, e.g. all txt or all jsonl
            types = [f.split('.')[-1] for f in eval_data_files]
            if len(set(types)) > 1:
                raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")  # 确保验证文件格式一致
        extension = "text" if data_files["train"][0].endswith('txt') else 'json'  # 根据扩展名决定加载方式
        if extension == "text":
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks  # 是否保留换行符
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
    logger.info(f"Raw datasets: {raw_datasets}")  # 输出原始数据集对象信息

    # Preprocessing the datasets.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)  # 获取训练集的列名（通常包含 text）
    else:
        column_names = list(raw_datasets["validation"].features)  # 若只评估，取验证集列名

    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        if not data_args.streaming:
            if training_args.group_by_length:
                tokenized_datasets = raw_datasets.map(
                    tokenize_wo_pad_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset" if is_main_process else None,
                )
                lm_datasets = tokenized_datasets.map(
                    group_text_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset" if is_main_process else None,
                )  # 得到的 lm_datasets 每条数据均为定长张量 (block_size)
        else:
            if training_args.group_by_length:
                tokenized_datasets = raw_datasets.map(
                    tokenize_wo_pad_function,
                    batched=True,
                    remove_columns=column_names,
                )
                lm_datasets = tokenized_datasets.map(
                    group_text_function,
                    batched=True,
                )
            else:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

    train_dataset = None  # 初始化训练集变量
    max_train_samples = 0  # 记录训练样本数量
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']  # 获取训练集 Dataset
        max_train_samples = len(train_dataset)  # 训练样本总数
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)  # 取较小值确保不超过设定上限
            train_dataset = train_dataset.select(range(max_train_samples))  # 仅保留前 max_train_samples 条样本
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))  # 输出第一条样本对应文本，检查 token 化效果

    eval_dataset = None  # 初始化验证集变量
    max_eval_samples = 0  # 记录验证样本数
    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )  # 将字符串 dtype 转为 torch.dtype 对象
        world_size = int(os.environ.get("WORLD_SIZE", "1"))  # 读取分布式进程总数，默认为 1（单机单卡）
        ddp = world_size != 1  # 判断是否处于分布式训练
        if ddp:
            model_args.device_map = {"": training_args.local_rank}  # 为每个进程分配对应 GPU
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")  # 提示 QLoRA 与部分并行策略不兼容

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }  # 传递给 AutoConfig 的关键词参数
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)  # 加载模型配置
        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")  # 不能同时使用 4bit 和 8bit
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
                else:
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            **config_kwargs,
        )  # 加载预训练模型，若启用量化则内部会根据 quantization_config 处理
    else:
        raise ValueError(f"Error, model_name_or_path is None, Continue PT must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)  # 加载已有 LoRA 权重继续训练
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)  # 量化模型需要额外准备
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None  # 解析目标模块
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)  # 自动扫描全部线性层
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
                # Resize the embedding layer to match the new tokenizer
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))  # 如果词表扩充，调整嵌入层大小
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)  # 在模型指定模块注入 LoRA 层
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)  # 将可训练参数强制转换为 float32，避免数值不稳定
        model.print_trainable_parameters()  # 输出可训练参数规模
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()  # 全参微调时确保参数为 float32
        print_trainable_parameters(model)

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存
        model.config.use_cache = False  # 为了梯度检查点，需要关闭缓存
    else:
        model.config.use_cache = True  # 训练策略允许时开启缓存，加速推理
    model.enable_input_require_grads()  # 确保输入嵌入保留梯度，支持 LoRA 训练
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True  # 通知 Trainer 模型支持手动并行
        model.model_parallel = True

    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )  # 初始化 Trainer，传入模型、数据、评估函数等

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")  # 打印一个 batch 样本结构
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint  # 若指定 checkpoint，则从断点继续
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  # 启动训练，返回 TrainOutput

        metrics = train_result.metrics  # 训练指标（loss、learning_rate 等）
        metrics["train_samples"] = max_train_samples  # 记录训练样本数
        trainer.log_metrics("train", metrics)  # 打印训练指标
        trainer.save_metrics("train", metrics)  # 保存到 JSON 文件
        trainer.save_state()  # 保存训练状态（优化器等）

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side  # 训练结束后恢复 padding_side（默认为左侧）
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)  # DeepSpeed ZeRO-3 使用特殊保存方式
            else:
                save_model(model, tokenizer, training_args)  # 常规保存方式

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()  # 在验证集上评估，返回指标字典

        metrics["eval_samples"] = max_eval_samples  # 记录验证样本数
        try:
            perplexity = math.exp(metrics["eval_loss"])  # 困惑度 = e^(交叉熵损失)
        except OverflowError:
            perplexity = float("inf")  # 若损失非常大导致溢出，则返回 inf
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)  # 打印评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估结果
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()  # 当脚本直接运行时执行 main()
