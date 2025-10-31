"""Accelerate SFT训练脚本"""  # 顶部文档字符串，说明该文件实现加速版监督微调流程

# -*- coding: utf-8 -*-  # 指定源码使用UTF-8编码，保证中文注释不会乱码
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 版权声明，表明使用Apache-2.0协议
# you may not use this file except in compliance with the License.  # 告知使用者必须遵守许可证条款
# You may obtain a copy of the License at  # 提供许可证获取方式
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache-2.0协议的官方链接
#
# Unless required by applicable law or agreed to in writing, software  # 协议条款说明无担保
# distributed under the License is distributed on an "AS IS" BASIS,  # 软件按“现状”提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何形式的担保
# See the License for the specific language governing permissions and  # 引导查看具体许可条款
# limitations under the License.  # 提醒存在的限制

import math  # 导入数学库，用于计算困惑度perplexity等指数运算
import os  # 导入操作系统库，用于环境变量和路径处理
import sys  # 导入sys以读取命令行参数以及脚本退出
from dataclasses import dataclass, field  # dataclass提供类自动生成__init__等方法；field用于字段默认值
from glob import glob  # glob用于匹配目录下满足模式的文件列表
from typing import Literal, Optional, Tuple  # 类型提示工具，Literal限定字符串取值范围

import torch  # 导入PyTorch核心库，用于张量运算与模型训练
import torch.utils.data  # 导入数据相关工具，如DataLoader
from datasets import load_dataset  # HuggingFace datasets加载器，用于读取数据集
from loguru import logger  # 引入loguru库用于结构化日志输出
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training  # 引入PEFT相关API，用于配置LoRA及低比特训练
from transformers import (  # 引入transformers库中模型与训练所需组件
    AutoConfig,  # 自动加载模型配置
    AutoModelForCausalLM,  # 自动加载因果语言模型
    AutoTokenizer,  # 自动加载分词器
    HfArgumentParser,  # 用于解析命令行参数为dataclass
    Seq2SeqTrainingArguments,  # 复用Seq2Seq训练参数容器（即使是CausalLM也可使用）
    set_seed,  # transformers提供的随机种子设置函数（虽然本脚本主要使用accelerate提供的版本）
    BitsAndBytesConfig,  # 量化配置类，用于4bit/8bit量化
    DataCollatorForSeq2Seq,  # 数据整理器，负责动态padding与label对齐
    get_linear_schedule_with_warmup,  # 学习率调度器，线性预热+线性下降
)
from transformers.trainer_pt_utils import LabelSmoother  # 引入标签平滑工具，用于忽略pad位置的loss
from tqdm.auto import tqdm  # 引入tqdm自动选择前端，提供训练进度条显示

from accelerate import Accelerator  # 引入Accelerator，用于封装分布式训练流程
from accelerate.utils import set_seed as accelerate_set_seed  # 从accelerate中导入专用的随机种子设置函数

is_flash_attn_2_available = False  # 初始化标志位，跟踪是否成功导入FlashAttention-2
try:  # 尝试导入FlashAttention-2相关函数
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # 如果可用，导入主要加速函数
    from flash_attn.bert_padding import pad_input, unpad_input  # 导入padding工具以适配变长输入

    is_flash_attn_2_available = True  # 导入成功则标记为可用状态
except ImportError:  # 如果导入失败
    is_flash_attn_2_available = False  # 保持标志位为False，后续逻辑会跳过相关优化
from template import get_conv_template  # 导入对话模板构造函数，用于将多轮对话转成模型输入


@dataclass  # 使用dataclass自动生成参数类的初始化函数
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""  # 官方原始注释，概述模型相关参数
    model_name_or_path: Optional[str] = field(default=None)  # 指定基础模型权重路径或huggingface模型名
    load_in_8bit: bool = field(default=False)  # 是否以8bit量化模式加载模型，节省显存
    load_in_4bit: bool = field(default=False)  # 是否以4bit量化模式加载模型，进一步压缩显存
    tokenizer_name_or_path: Optional[str] = field(default=None)  # 单独指定分词器路径；为空则复用模型路径
    cache_dir: Optional[str] = field(default=None)  # 指定模型/数据缓存目录
    model_revision: Optional[str] = field(default="main")  # HuggingFace模型版本标签，例如"main"或具体commit
    hf_hub_token: Optional[str] = field(default=None)  # 访问私有模型仓库时使用的HF令牌
    use_fast_tokenizer: bool = field(default=False)  # 是否使用基于tokenizers库的高速分词器
    torch_dtype: Optional[str] = field(default="float16")  # 读取模型时的默认dtype，字符串在下游转换为torch dtype
    device_map: Optional[str] = field(default="auto")  # 当启用device_map时指定映射策略，例如"auto"
    trust_remote_code: bool = field(default=True)  # 是否信任远程仓库自定义代码（必要时启用）
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)  # 指定RoPE位置编码缩放策略
    flash_attn: Optional[bool] = field(  # 控制是否尝试启用FlashAttention-2
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}  # CLI帮助文本
    )


@dataclass  # 使用dataclass承载数据相关参数
class DataArguments:
    dataset_name: Optional[str] = field(default=None,
                                        metadata={"help": "The name of the dataset to use (via the datasets library)."})  # 若指定则从HF Hub加载数据集
    dataset_config_name: Optional[str] = field(default=None, metadata={
        "help": "The configuration name of the dataset to use (via the datasets library)."})  # 指定数据集配置子集
    train_file_dir: str = field(default=None, metadata={"help": "Path to the training data."})  # 本地训练数据文件目录
    validation_file_dir: str = field(default=None, metadata={"help": "Path to the validation data."})  # 本地验证数据目录
    max_train_samples: Optional[int] = field(default=None)  # 限制训练样本数量，None表示使用全部
    max_eval_samples: Optional[int] = field(default=None)  # 限制验证样本数量
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})  # 是否重建缓存
    validation_split_percentage: Optional[int] = field(default=1)  # 当没有验证集时，从训练集中切分百分比
    preprocessing_num_workers: Optional[int] = field(default=None)  # 指定map函数的并行进程数
    ignore_pad_token_for_loss: bool = field(default=True)  # 是否在loss计算时忽略pad标记


@dataclass  # 额外脚本行为参数
class ScriptArguments:
    use_peft: bool = field(default=True)  # 是否启用PEFT (LoRA/QLoRA) 微调模式
    train_on_inputs: bool = field(default=False)  # 是否让输入token也参与loss（一般SFT忽略输入梯度）
    target_modules: Optional[str] = field(default="all")  # LoRA注入的目标模块列表，逗号分隔；all表示自动发现
    lora_rank: Optional[int] = field(default=8)  # LoRA秩r，控制额外矩阵的秩
    lora_dropout: Optional[float] = field(default=0.05)  # LoRA Dropout比例，防过拟合
    lora_alpha: Optional[float] = field(default=32.0)  # LoRA缩放因子alpha
    modules_to_save: Optional[str] = field(default=None)  # 指定除LoRA外需要保存的模块，例如embed_tokens
    peft_path: Optional[str] = field(default=None)  # 若提供已有LoRA权重路径，则基于其继续训练
    qlora: bool = field(default=False)  # 是否使用QLoRA（配合4bit量化）
    model_max_length: int = field(default=2048)  # 单条样本最大token长度，用于截断
    template_name: Optional[str] = field(default="vicuna")  # 选择对话模板名称，确保训练推理一致
    use_tensor_parallel: bool = field(  # 是否启用张量并行（模型分片到多个GPU）
        default=False,
        metadata={"help": "Whether to use tensor parallelism for large models"}  # CLI帮助文本
    )


def find_all_linear_names(model, int4=False, int8=False):
    """查找模型中所有的线性层名称"""  # 返回模型内可注入LoRA的线性层名称集合
    cls = torch.nn.Linear  # 默认匹配PyTorch标准Linear层
    if int4 or int8:  # 如果使用了低比特量化
        import bitsandbytes as bnb  # 延迟导入bitsandbytes，避免非量化情况下的依赖
        if int4:  # 4bit量化时
            cls = bnb.nn.Linear4bit  # 切换匹配类到4bit线性层
        elif int8:  # 8bit量化时
            cls = bnb.nn.Linear8bit  # 切换匹配类到8bit线性层
    lora_module_names = set()  # 初始化集合存储符合条件的模块名称
    for name, module in model.named_modules():  # 遍历模型中的全部子模块，name形如"model.layers.0.self_attn.q_proj"
        if isinstance(module, cls):  # 仅处理与目标Linear类匹配的模块
            if 'lm_head' in name:  # 跳过输出层lm_head，避免在最终投影层注入LoRA
                continue
            if 'output_layer' in name:  # 某些模型自定义的输出层同样跳过
                continue
            names = name.split('.')  # 将层名称按"."拆分，例如['model','layers','0','self_attn','q_proj']
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  # 取最后一级模块名称登记集合
    return sorted(lora_module_names)  # 返回排序后的模块列表，方便可重复使用


def save_model(model, tokenizer, output_dir):
    """Save the model and the tokenizer."""  # 保存模型与分词器到指定目录
    os.makedirs(output_dir, exist_ok=True)  # 若输出目录不存在则创建，exist_ok避免重复报错

    model_to_save = model.module if hasattr(model, "module") else model  # 若模型被DDP包裹，取其内部原始模型
    model_to_save.save_pretrained(output_dir)  # 以HuggingFace格式保存权重和配置
    tokenizer.save_pretrained(output_dir)  # 保存分词器词表及相关配置


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""  # 输出模型可训练参数占比
    trainable_params = 0  # 初始化可训练参数计数
    all_param = 0  # 初始化全部参数计数
    for _, param in model.named_parameters():  # 遍历所有参数张量
        all_param += param.numel()  # numel()返回张量元素个数，累加得到总参数量
        if param.requires_grad:  # 仅统计需要梯度的参数
            trainable_params += param.numel()
    print(  # 通过标准输出打印结果，包含总量、可训练量、百分比
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_datasets(data_args, model_args):
    """Load datasets from files or HuggingFace hub"""  # 根据传参决定数据集来源
    if data_args.dataset_name is not None:  # 若提供数据集名称，从HF Hub下载
        raw_datasets = load_dataset(  # load_dataset会返回DatasetDict，包含train/validation等split
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():  # 若缺少validation划分
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)  # 先对train随机打乱，保证划分均匀
            split = shuffled_train_dataset.train_test_split(  # 利用datasets内置函数切分
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            raw_datasets["train"] = split["train"]  # 将切分后的训练部分写回train
            raw_datasets["validation"] = split["test"]  # 切分出的测试部分作为validation
    else:  # 否则从本地JSON/JSONL文件加载
        data_files = {}  # 初始化文件路径字典
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):  # 判断训练数据目录是否存在
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)  # 递归匹配所有json/jsonl文件
            logger.info(f"train files: {train_data_files}")  # 打印匹配文件列表
            data_files["train"] = train_data_files  # 加入train键
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):  # 同理验证集
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(  # 以json加载器读取本地多个文件合并为单个Dataset
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():  # 若仍然没有validation split
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)  # 先随机打乱训练集
            split = shuffled_train_dataset.train_test_split(  # 依照比例切分
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]  # 更新train集合
            raw_datasets["validation"] = split["test"]  # 更新validation集合

    logger.info(f"Raw datasets: {raw_datasets}")  # 记录最终数据集概况
    return raw_datasets  # 返回DatasetDict供后续使用


def create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """Create preprocessing function for datasets"""  # 返回闭包，用于将原始对话样本转为模型输入
    max_length = script_args.model_max_length  # 从脚本参数读取全局最大token长度，用于后续截断

    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """  # 内部函数按datasets.map接口处理一个批次examples
        input_ids_list = []  # 保存每条样本的token序列，形状为[List[int]], 每条长度<=max_length
        attention_mask_list = []  # 保存attention_mask，与input_ids等长，1表示有效token
        targets_list = []  # 保存labels，忽略输入部分用IGNORE_INDEX占位
        roles = ["human", "gpt"]  # 预期角色顺序，人类提问->模型回答

        def get_dialog(examples):  # 内部生成器，将原始conversation结构展开成字符串序列
            system_prompts = examples.get("system_prompt", "")  # 尝试读取批次中的系统提示列表，可能为空
            for i, source in enumerate(examples['conversations']):  # 遍历每条样本的conversation字段
                system_prompt = ""  # 默认系统提示为空
                if len(source) < 2:  # 对话轮次不足2（缺少问答对）则跳过
                    continue
                data_role = source[0].get("from", "")  # 读取第一条消息的角色
                if data_role == "system":  # 第一条如果是系统消息
                    system_prompt = source[0]["value"]  # 保存系统提示文本
                    source = source[1:]  # 将后续对话作为真正的交互
                    data_role = source[0].get("from", "")  # 更新第一条消息的角色
                if data_role not in roles or data_role != roles[0]:  # 如果第一条不是human角色
                    source = source[1:]  # 跳过这一条，尝试从下一条开始
                if len(source) < 2:  # 再次检查剩余长度，确保至少一对问答
                    continue
                messages = []  # 存储按顺序排列的消息内容
                for j, sentence in enumerate(source):  # 遍历剩余的消息
                    data_role = sentence.get("from", "")  # 当前消息的角色
                    if data_role not in roles:  # 发现未知角色
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")  # 记录警告并忽略整条样本
                        break
                    if data_role == roles[j % 2]:  # 验证角色顺序是否与human/gpt交替匹配
                        messages.append(sentence["value"])  # 收集消息文本
                if len(messages) % 2 != 0:  # 若消息数量不是偶数（问答不成对）则跳过
                    continue
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]  # 将列表拆成[[问,答], ...]
                if not system_prompt:  # 如果当前样本没写系统提示
                    system_prompt = system_prompts[i] if system_prompts else ""  # 尝试从批次字段补齐
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)  # 使用模板转换为扁平化字符串序列

        for dialog in get_dialog(examples):  # 遍历生成的模板化dialog，dialog为长度2n的列表[首轮prompt,首轮答复,...]
            input_ids, labels = [], []  # 初始化该样本的输入序列与标签序列

            for i in range(len(dialog) // 2):  # 每次处理一对问答（索引i对应第i轮对话）
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))  # 将提问编码为token列表，首轮允许加入BOS等特殊符号
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)  # 将回答编码为token列表，不添加额外符号

                total_len = len(source_ids) + len(target_ids)  # 当前轮次问答的总token数，用于按比例分配截断长度
                max_source_len = int(max_length * (len(source_ids) / total_len))  # 计算source可占用长度，形如floor(max_len * source_ratio)
                max_target_len = int(max_length * (len(target_ids) / total_len))  # 同理，计算target允许长度

                if len(source_ids) > max_source_len:  # 若提问超过分配的长度
                    source_ids = source_ids[:max_source_len]  # 截断source_ids，形状保持1D
                if len(target_ids) > max_target_len - 1:  # 预留1个位置给EOS token
                    target_ids = target_ids[:max_target_len - 1]  # 截断回答token
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:  # 处理编码时可能开头是EOS的情况
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:  # 若回答末尾已有EOS，则移除避免重复
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:  # 检查累计长度+1（追加EOS）是否超限
                    break  # 超过上限则停止添加后续轮次

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # 将问答及结尾EOS拼接进输入，序列长度更新为旧值+len(src)+len(tgt)+1
                if script_args.train_on_inputs:  # 若训练时包含输入tokens在loss中
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]  # labels与input_ids完全对齐
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]  # 输入部分用IGNORE_INDEX填充，忽略梯度

            input_ids_list.append(input_ids)  # 追加当前样本的token序列，形状: [seq_len]
            attention_mask_list.append([1] * len(input_ids))  # attention_mask与input_ids等长，全部为1（后续collator会pad）
            targets_list.append(labels)  # 追加标签序列，长度与input_ids一致

        return dict(  # 返回datasets要求的字典，键为字段名，值为批量数据列表
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    return preprocess_function  # 返回闭包供map调用


def filter_empty_labels(example, IGNORE_INDEX):
    """Remove empty labels dataset."""  # 过滤掉labels全为IGNORE_INDEX的样本，避免无监督信号
    return not all(label == IGNORE_INDEX for label in example["labels"])  # 如果存在至少一个真实标签则保留


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""  # 打印当前GPU内存使用情况并启用可用的高效注意力实现
    if not torch.cuda.is_available():  # 若无GPU则无需处理
        return

    logger.info("🔍 检查GPU内存状态...")  # 提示开始检查

    torch.cuda.empty_cache()  # 清空PyTorch缓存，释放未使用显存

    num_gpus = torch.cuda.device_count()  # 获取GPU数量
    for i in range(num_gpus):  # 遍历每张GPU
        props = torch.cuda.get_device_properties(i)  # 获取硬件属性
        total_memory = props.total_memory / 1024 ** 3  # 计算总显存，单位GB
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # 当前已分配显存GB
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3  # PyTorch缓存显存GB
        free = total_memory - allocated - cached  # 估算可用显存GB

        logger.info(f"GPU {i} ({props.name}):")  # 输出GPU名称
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

        if free < 2.0:  # 如果可用显存低于2GB，提示优化策略
            logger.warning(f"⚠️ GPU {i} 可用内存不足 ({free:.1f}GB)，建议:")
            logger.warning("  1. 使用 --load_in_4bit 启用4bit量化")
            logger.warning("  2. 减小 --per_device_train_batch_size")
            logger.warning("  3. 增加 --gradient_accumulation_steps")
            logger.warning("  4. 减小 --model_max_length")

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):  # 检查是否支持Flash SDP实现
        torch.backends.cuda.enable_flash_sdp(True)  # 启用FlashAttention内核
        logger.info("✅ 启用Flash Attention优化")

    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):  # 检查是否支持内存高效注意力
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def get_unwrapped_model(model):
    """获取未包装的原始模型，无论它是否被DDP包装"""  # 统一返回基础模型对象
    if hasattr(model, "module"):  # 当模型被DDP/DataParallel包装时
        return model.module  # 取内部module
    return model  # 否则直接返回


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 设置CUDA显存分配策略，允许弹性分段分配
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))  # 建立参数解析器，自动映射到四个dataclass

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):  # 若命令行只给一个json文件，则按json配置解包
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))  # 将json路径转绝对路径后解析为四个参数对象
    else:  # 否则直接从命令行解析
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)  # parse_args_into_dataclasses返回对应参数实例

    logger.info(f"🚀 使用Accelerate库进行多GPU训练")  # 日志记录训练模式
    logger.info("🚀 开始初始化Accelerator...")  # 提示Accelerator初始化
    accelerator = Accelerator()  # 创建Accelerator实例，内部处理分布式/混合精度等配置
    logger.info("✅ Accelerator初始化完成")  # 标记初始化完成
    try:  # 尝试打印更多状态信息
        logger.info(f"设备: {accelerator.device}")  # 输出当前设备
        logger.info(f"检测到 {accelerator.num_processes} 个进程")  # 输出分布式进程数
        logger.info(f"当前进程: {accelerator.process_index}")  # 输出当前rank
        logger.info(f"分布式类型: {accelerator.distributed_type}")  # 输出分布式后端信息
    except Exception:  # 捕获可能的属性访问异常
        logger.warning("无法获取完整的Accelerator信息，但这不影响训练")  # 打印警告但不中断

    logger.info(f"Model args: {model_args}")  # 记录模型相关参数
    logger.info(f"Training args: {training_args}")  # 记录训练参数
    logger.info(f"Script args: {script_args}")  # 记录脚本自定义参数

    accelerate_set_seed(training_args.seed)  # 设置全局随机种子，确保结果可复现

    tokenizer_kwargs = {  # 构造加载分词器的关键字参数
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path  # 优先使用单独指定的分词器路径，否则复用模型路径
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)  # 加载分词器对象

    prompt_template = get_conv_template(script_args.template_name)  # 获取指定名称的对话模板，用于补充特殊token并格式化输入
    if tokenizer.eos_token_id is None:  # 若分词器尚未定义EOS
        tokenizer.eos_token = prompt_template.stop_str  # 使用模板提供的停用符作为EOS
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})  # 注册EOS token
        logger.info(f"Add eos_token: {tokenizer.eos_token}")  # 记录新增token

    if tokenizer.bos_token_id is None:  # 若缺少BOS
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})  # 将EOS复用为BOS（不少开源模型这样处理）
        tokenizer.bos_token_id = tokenizer.eos_token_id  # 同步BOS ID
        logger.info(f"Add bos_token: {tokenizer.bos_token}")

    if tokenizer.pad_token_id is None:  # 若缺少PAD
        if tokenizer.unk_token_id is not None:  # 优先使用UNK作为PAD
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token  # 否则退化为使用EOS作为PAD
        logger.info(f"Add pad_token: {tokenizer.pad_token}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id  # 根据配置确定loss中忽略的标签值

    logger.info("✅ Tokenizer配置完成")  # 标记分词器准备完毕

    check_and_optimize_memory()  # 在加载模型前先释放显存并启用可选优化

    logger.info("🔄 开始加载模型...")  # 提示模型加载阶段

    torch_dtype = model_args.torch_dtype  # 记录期望的模型dtype（字符串形式，例如"float16"）
    quantization_config = None  # 默认不使用量化配置
    if model_args.load_in_4bit:  # 如需4bit量化
        quantization_config = BitsAndBytesConfig(  # 创建4bit量化配置
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,  # 计算时使用的dtype，如float16/bfloat16
            bnb_4bit_use_double_quant=True,  # 启用double quant减小量化误差
            bnb_4bit_quant_type="nf4"  # 指定NF4量化类型
        )
    elif model_args.load_in_8bit:  # 如需8bit量化
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 创建8bit量化配置

    config_kwargs = {  # 准备加载模型配置的参数
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "hf_hub_token": model_args.hf_hub_token,
    }
    if model_args.flash_attn:  # 如果用户要求启用FlashAttention
        if is_flash_attn_2_available:  # 检查是否已成功导入
            config_kwargs["use_flash_attention_2"] = True  # 在配置中打开FlashAttention标志
            logger.info("Using FlashAttention-2 for faster training and inference.")  # 记录启用信息
        else:
            logger.warning("FlashAttention-2 is not installed.")  # 提醒缺少依赖
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)  # 加载模型配置对象

    total_memory = 0  # 初始化GPU总显存统计
    if torch.cuda.is_available():  # 仅在GPU环境下执行
        num_gpus = torch.cuda.device_count()  # 获取GPU数量
        logger.info(f"检测到 {num_gpus} 个GPU")  # 打印GPU数量

        for i in range(num_gpus):  # 遍历每块GPU
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # 单卡总显存GB
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # 该卡已分配显存GB
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3  # 该卡缓存显存GB
            free = gpu_memory - allocated  # 估算未分配显存（忽略缓存影响）
            total_memory += gpu_memory  # 累加总显存
            logger.info(  # 输出每卡详细信息
                f"GPU {i}: 总内存={gpu_memory:.1f}GB, 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 可用={free:.1f}GB")

        logger.info(f"总GPU内存: {total_memory:.1f}GB")  # 输出显存总量

        torch.cuda.empty_cache()  # 再次清理缓存，保证后续加载充足显存
        logger.info("已清理GPU缓存")  # 记录操作

    estimated_model_size_gb = 0  # 初始化模型大小估计值
    if hasattr(config, 'num_parameters'):  # 若配置文件提供参数总数
        estimated_model_size_gb = config.num_parameters * 2 / 1024 ** 3  # 按fp16假设换算为GB (参数量 *2 bytes / 1024^3)
    else:
        model_name_lower = model_args.model_name_or_path.lower()  # 将模型名转小写，便于匹配
        if '70b' in model_name_lower or '72b' in model_name_lower:
            estimated_model_size_gb = 140  # 70B等级模型约140GB(fp16)
        elif '32b' in model_name_lower or '34b' in model_name_lower:
            estimated_model_size_gb = 64
        elif '13b' in model_name_lower or '14b' in model_name_lower:
            estimated_model_size_gb = 26
        elif '7b' in model_name_lower or '8b' in model_name_lower:
            estimated_model_size_gb = 14
        elif '3b' in model_name_lower:
            estimated_model_size_gb = 6
        else:
            estimated_model_size_gb = 10  # 默认给出保守估计

    logger.info(f"估算模型大小: {estimated_model_size_gb:.1f}GB")  # 输出估算结果

    num_gpus = torch.cuda.device_count()  # 再次记录GPU数量供后续逻辑
    is_distributed = accelerator.num_processes > 1  # 是否处于多进程训练

    if is_distributed:  # 多进程训练下
        if script_args.use_tensor_parallel and estimated_model_size_gb > 20:  # 若允许张量并行且模型较大
            logger.info(f"🔧 使用张量并行策略 (模型大小: {estimated_model_size_gb:.1f}GB)")
            use_tensor_parallel = True  # 默认启用张量并行

            import pkg_resources  # 延迟导入pkg_resources用于版本比较
            torch_version = pkg_resources.get_distribution("torch").version  # 读取当前PyTorch版本
            if pkg_resources.parse_version(torch_version) < pkg_resources.parse_version("2.5.0"):  # 版本不足
                logger.warning(f"⚠️ 当前PyTorch版本 {torch_version} 不支持张量并行，需要 >= 2.5.0")
                logger.warning("⚠️ 自动切换到DDP模式")
                use_tensor_parallel = False  # 回退到DDP
            else:
                logger.info(f"✅ PyTorch版本 {torch_version} 支持张量并行")
        else:
            logger.info(f"🔧 使用DDP进行多GPU训练 (模型大小: {estimated_model_size_gb:.1f}GB)")
            use_tensor_parallel = False  # 默认使用传统DDP
    else:
        logger.info("🔧 单进程训练")  # 单机单进程时
        use_tensor_parallel = True  # 允许device_map自动切分（也可能只是单卡）

    model_kwargs = {  # 准备加载模型的关键字参数
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,  # 使用低CPU内存加载方式
    }

    if use_tensor_parallel:  # 当使用device_map切分模型
        model_kwargs["device_map"] = "auto"  # 自动根据显存切分模块

        if num_gpus > 1:  # 多GPU时设置每卡最大显存限制
            max_memory = {}
            for i in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)  # 获取显卡属性
                total_mem = gpu_props.total_memory  # 总显存（字节）
                usable_mem = int(total_mem * 0.8)  # 预留20%缓冲
                max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"  # 转换为字符串表示，例如"60GiB"

            model_kwargs["max_memory"] = max_memory  # 传入max_memory字典限制占用
            logger.info(f"🔧 张量并行配置:")
            logger.info(f"  device_map: auto")
            logger.info(f"  max_memory: {max_memory}")
    else:
        logger.info("🔧 DDP配置: 不使用device_map")  # DDP模式直接由Accelerate负责参数分发

    try:  # 尝试加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("✅ 模型加载完成")
    except OSError as e:  # 处理device_map不被支持的情况
        if "tensor parallel is only supported for" in str(e):  # 特定报错指示张量并行不被支持
            logger.error(f"❌ 张量并行加载失败: {e}")
            logger.info("🔄 尝试使用DDP模式重新加载...")
            if "device_map" in model_kwargs:
                del model_kwargs["device_map"]  # 移除device_map约束
            if "max_memory" in model_kwargs:
                del model_kwargs["max_memory"]  # 移除显存限制

            model = AutoModelForCausalLM.from_pretrained(  # 再次加载模型
                model_args.model_name_or_path,
                **model_kwargs
            )
            logger.info("✅ 使用DDP模式加载模型成功")
        else:
            raise  # 其他错误直接抛出

    logger.info("📊 模型分布情况:")  # 输出模型设备分布
    if hasattr(model, 'hf_device_map') and model.hf_device_map:  # 如果模型自带device_map说明已分片
        logger.info("🔧 使用HuggingFace设备映射:")
        for module_name, device in model.hf_device_map.items():  # 遍历每个模块所在设备
            logger.info(f"  {module_name}: {device}")

        device_count = {}  # 统计每个设备承载模块数量
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_count[device_str] = device_count.get(device_str, 0) + 1

        logger.info("📈 设备使用统计:")
        for device, count in device_count.items():
            logger.info(f"  {device}: {count} 个模块")
    else:
        device_params = {}  # 字典跟踪各设备参数数量
        total_params = 0  # 全部参数数量
        for name, param in model.named_parameters():  # 遍历每个可训练参数
            device = str(param.device)
            if device not in device_params:
                device_params[device] = {'count': 0, 'size': 0}
            device_params[device]['count'] += 1
            device_params[device]['size'] += param.numel()  # 累加参数元素数
            total_params += param.numel()

        logger.info("📈 参数设备分布:")
        for device, info in device_params.items():
            param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32（4字节）估算占用
            percentage = info['size'] / total_params * 100  # 计算百分比
            logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

    if torch.cuda.is_available():  # 额外打印当前显存占用
        logger.info("💾 GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # 已分配显存GB
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3  # 缓存显存GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # 总显存GB
            logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

    if script_args.use_peft:  # 根据配置决定是否注入LoRA
        logger.info("🔧 配置LoRA")  # 日志提示进入LoRA设置

        if script_args.peft_path is not None:  # 若提供已有LoRA路径
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)  # 加载LoRA权重并保持可训练
        else:
            if model_args.load_in_8bit or model_args.load_in_4bit:  # 如果模型为低比特量化
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)  # 调整层的dtype和梯度配置以适应k-bit训练

            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None  # 解析用户传入的目标模块
            if target_modules and 'all' in target_modules:  # all表示自动发现所有线性层
                target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit,
                                                       int8=model_args.load_in_8bit)

            modules_to_save = script_args.modules_to_save  # 读取需要单独保存的模块
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')  # 转为列表

            peft_config = LoraConfig(  # 构建LoRA配置
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)  # 将LoRA适配器注入模型

        for param in filter(lambda p: p.requires_grad, model.parameters()):  # 遍历所有需要梯度的参数
            param.data = param.data.to(torch.float32)  # 保证LoRA权重使用fp32存储，提高数值稳定性

        model.print_trainable_parameters()  # 打印LoRA后可训练参数统计
    else:
        logger.info("🔧 全参数训练模式")  # 不使用LoRA时进行全量微调
        model = model.float()  # 将模型全部转换到float32以稳定训练
        print_trainable_parameters(model)  # 输出全参数训练的参数规模

    logger.info("🔄 开始加载数据集...")  # 标记数据加载阶段
    raw_datasets = load_datasets(data_args, model_args)  # 根据参数解析出的路径或Hub名称读取DatasetDict

    logger.info("🔄 开始预处理数据集...")  # 提示预处理开始
    preprocess_function = create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX)  # 构造datasets.map使用的预处理函数

    train_dataset = None  # 初始化训练集合
    max_train_samples = 0  # 记录实际使用的训练样本数
    if training_args.do_train:  # 如果需要训练
        if "train" not in raw_datasets:  # 检查train split是否存在
            raise ValueError("--do_train requires a train dataset")  # 缺少训练集时立即报错提示
        train_dataset = raw_datasets['train'].shuffle(seed=42)  # 打乱训练集，返回新的Dataset对象
        max_train_samples = len(train_dataset)  # 记录初始样本数
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:  # 如果指定了采样上限
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)  # 更新最大样本数
            train_dataset = train_dataset.select(range(max_train_samples))  # 选择前N个样本，返回新的Dataset

        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")  # 打印示例样本结构

        tokenized_dataset = train_dataset.map(  # 调用map批量处理，输出字段为input_ids/attention_mask/labels
            preprocess_function,  # 指定刚才构造的预处理函数
            batched=True,  # 启用批处理模式，加速tokenization
            num_proc=data_args.preprocessing_num_workers,  # 并行worker数量，提高吞吐
            remove_columns=train_dataset.column_names,  # 删除原始列，只保留新生成字段
            load_from_cache_file=not data_args.overwrite_cache,  # 默认使用缓存以避免重复计算
            desc="Running tokenizer on dataset",  # 进度条描述
        )
        train_dataset = tokenized_dataset.filter(  # 过滤掉空标签样本
            lambda example: filter_empty_labels(example, IGNORE_INDEX),  # 针对每条样本执行过滤逻辑
            num_proc=data_args.preprocessing_num_workers  # 并行worker数量与map保持一致
        )

        logger.debug(f"Num train_samples: {len(train_dataset)}")  # 输出过滤后样本数
        logger.debug("Tokenized training example:")  # 打印提示，以下展示token化后的样例
        logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")  # 解码input_ids查看内容
        replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                           for label in list(train_dataset[0]['labels'])]  # 将IGNORE_INDEX替换为pad token便于解码预览
        logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    eval_dataset = None  # 初始化验证集合
    max_eval_samples = 0  # 记录验证样本数量
    if training_args.do_eval:  # 如果需要评估
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")  # 无验证集时抛出异常
        eval_dataset = raw_datasets["validation"]  # 读取验证split
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_size = len(eval_dataset)
        logger.debug(f"Num eval_samples: {eval_size}")
        if eval_size > 500:
            logger.warning(f"Num eval_samples is large: {eval_size}, training slow, consider reduce it by `--max_eval_samples=50`")  # 提醒验证集过大可能拖慢训练
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")  # 打印验证集样例
        eval_dataset = eval_dataset.map(  # 对验证集执行相同token化步骤
            preprocess_function,  # 复用训练阶段的预处理逻辑
            batched=True,  # 按批次预处理
            num_proc=data_args.preprocessing_num_workers,  # 并行worker数量
            remove_columns=eval_dataset.column_names,  # 移除原始列
            load_from_cache_file=not data_args.overwrite_cache,  # 默认使用缓存
            desc="Running tokenizer on validation dataset",  # 设定进度条文字
        )
        eval_dataset = eval_dataset.filter(
            lambda example: filter_empty_labels(example, IGNORE_INDEX),  # 同样过滤空标签样本
            num_proc=data_args.preprocessing_num_workers  # 可选多进程过滤
        )
    logger.debug(f"Num eval_samples: {len(eval_dataset)}")  # 输出token化后验证样本数量
    logger.debug("Tokenized eval example:")  # 提示接下来展示token化后的验证样本
    logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))  # 解码查看验证样本内容

    logger.info("✅ 数据集预处理完成")  # 标记数据准备完毕

    data_collator = DataCollatorForSeq2Seq(  # 初始化数据整理器
        tokenizer=tokenizer,  # 提供分词器以执行padding
        model=model,  # 提供模型以便collator能获取特殊参数（如prepare_decoder_input_ids）
        label_pad_token_id=IGNORE_INDEX,  # 标签padding值使用IGNORE_INDEX，保证loss忽略
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # 若右侧padding则按4对齐，利于张量核展开
    )

    train_dataloader = None  # 训练数据加载器默认空
    eval_dataloader = None  # 验证数据加载器默认空

    if training_args.do_train and train_dataset is not None:  # 当存在训练集
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,  # Dataset对象，内部按索引返回样本
            batch_size=training_args.per_device_train_batch_size,  # 每张卡的批大小
            shuffle=True,  # 按epoch随机打乱
            collate_fn=data_collator,  # 使用上面的collator完成padding和张量堆叠
        )  # DataLoader输出batch字典：input_ids(batch, seq_len)、attention_mask同形状、labels同形状

    if training_args.do_eval and eval_dataset is not None:  # 当存在验证集
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,  # 验证数据集
            batch_size=training_args.per_device_eval_batch_size,  # 验证阶段批大小
            shuffle=False,  # 验证无需打乱
            collate_fn=data_collator,  # 同一collator保证形状一致
        )

    optimizer = None  # 初始化优化器
    lr_scheduler = None  # 初始化学习率调度器

    if training_args.do_train:  # 仅在训练模式下配置优化器
        optimizer = torch.optim.AdamW(  # 使用AdamW优化LoRA或全参数权重
            filter(lambda p: p.requires_grad, model.parameters()),  # 仅更新需要梯度的参数（LoRA层）
            lr=training_args.learning_rate,  # 初始学习率
            weight_decay=training_args.weight_decay,  # L2正则
        )

        num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps  # 每个epoch中真实的参数更新步数= (steps_per_epoch / 累积步数)
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch  # 总更新步数=epoch数*每epoch更新次数

        lr_scheduler = get_linear_schedule_with_warmup(  # 构造线性warmup+线性衰减学习率策略
            optimizer=optimizer,  # 绑定优化器
            num_warmup_steps=int(max_train_steps * training_args.warmup_ratio),  # warmup步数=总步数*比例
            num_training_steps=max_train_steps,  # 计划的全部训练步数
        )

    logger.info("🔄 开始准备训练组件...")  # 进入Accelerate准备阶段

    model_is_distributed = hasattr(model, 'hf_device_map') and model.hf_device_map  # 判断模型是否已按device_map切分

    if model_is_distributed:  # 如果模型已经预分配到多个设备
        logger.info("🔧 检测到模型已分布在多设备，使用兼容模式")  # 日志提示当前采用兼容处理
        if training_args.do_train:  # 当执行训练时
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(  # 仅将优化器和dataloader交给accelerator
                optimizer, train_dataloader, lr_scheduler
            )
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        model.train() if training_args.do_train else model.eval()  # 手动设置模型mode

        logger.info("✅ 分布式模型训练组件准备完成")  # 记录兼容模式下准备完毕
    else:
        logger.info("🔧 标准模式，让Accelerate处理所有组件")  # 日志提示进入标准模式
        if training_args.do_train:
            model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(  # 让Accelerator同时包装模型+优化器+数据
                model, optimizer, train_dataloader, lr_scheduler
            )
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            model = accelerator.prepare(model)
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("✅ 标准训练组件准备完成")  # 标记标准模式准备完成

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):  # 条件启用梯度检查点，减少显存
        model.gradient_checkpointing_enable()
        if hasattr(model, "module"):
            model.module.config.use_cache = False  # 关闭KV缓存以兼容梯度检查点
            logger.info("Gradient checkpointing enabled for DDP model.")  # 日志说明DDP模型启用梯度检查点
        else:
            model.config.use_cache = False
            logger.info("Gradient checkpointing enabled.")  # 单卡模型启用梯度检查点
    else:
        if hasattr(model, "module"):
            model.module.config.use_cache = True  # 若未启用则恢复默认缓存
            logger.info("Gradient checkpointing disabled for DDP model.")  # 日志说明未启用梯度检查点
        else:
            model.config.use_cache = True
            logger.info("Gradient checkpointing disabled.")  # 单卡模型保持默认cache设置
    if hasattr(model, "module"):
        model.module.enable_input_require_grads()  # 确保输入embedding支持梯度（LoRA常用）
    else:
        model.enable_input_require_grads()

    logger.info("🎉 Accelerate多GPU训练配置成功！")

    if training_args.do_train:  # 训练主循环入口
        logger.info("*** 开始训练 ***")

        model.train()  # 切换到训练模式
        total_loss = 0  # 累计损失，用于日志平均
        completed_steps = 0  # 已完成的优化步数

        progress_bar = tqdm(  # 创建进度条，总步数=epoch数*每epoch迭代数
            range(int(training_args.num_train_epochs * len(train_dataloader))),  # 迭代上限序列，用于驱动进度显示
            disable=not accelerator.is_local_main_process,  # 非主进程关闭进度条，避免多卡重复输出
            desc="Training"  # 进度条标题
        )

        for epoch in range(int(training_args.num_train_epochs)):  # 外层循环遍历每个epoch
            logger.info(f"开始第 {epoch + 1}/{int(training_args.num_train_epochs)} 轮训练")

            for step, batch in enumerate(train_dataloader):  # batch是字典：input_ids(batch,seq)、attention_mask同shape、labels同shape
                if model_is_distributed:  # 当模型已手动分布时
                    outputs = model(**batch)  # 前向传播，batch张量默认已在正确设备
                    loss = outputs.loss  # 取出平均loss标量

                    if training_args.gradient_accumulation_steps > 1:
                        loss = loss / training_args.gradient_accumulation_steps  # 累积梯度时按比例缩放loss

                    loss.backward()  # 反向传播，计算梯度

                    if (step + 1) % training_args.gradient_accumulation_steps == 0:  # 达到一次真实优化步
                        if training_args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)  # 执行梯度裁剪

                        optimizer.step()  # 更新权重
                        lr_scheduler.step()  # 更新学习率
                        optimizer.zero_grad()  # 清空累积梯度

                        completed_steps += 1  # 记录完成的优化步数
                        progress_bar.update(1)  # 进度条前进一步
                else:
                    with accelerator.accumulate(model):  # Accelerate自动处理梯度累积
                        outputs = model(**batch)
                        loss = outputs.loss

                        accelerator.backward(loss)  # 使用Accelerate包装的反向传播，兼容混合精度

                        if accelerator.sync_gradients:  # 当达到累积步触发同步时
                            accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                        optimizer.step()  # 同步时执行权重更新
                        lr_scheduler.step()  # 更新学习率调度器
                        optimizer.zero_grad()  # 清空梯度缓存

                    if accelerator.sync_gradients:  # 仅在真实更新步时增加计数
                        completed_steps += 1  # 增加完成步数
                        progress_bar.update(1)  # 更新进度条

                total_loss += loss.detach().float()  # 累加当前loss（detach防梯度连锁）

                step_completed = False  # 标记是否完成真实更新步
                if model_is_distributed:
                    step_completed = (step + 1) % training_args.gradient_accumulation_steps == 0  # 张量并行情况下的判断条件
                else:
                    step_completed = accelerator.sync_gradients  # Accelerate模式下直接读取同步标志

                if step_completed:  # 仅在真实更新步执行日志、保存与评估
                    if completed_steps % training_args.logging_steps == 0:  # 日志记录间隔
                        avg_loss = total_loss / training_args.logging_steps  # 计算过去若干步的平均损失
                        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else training_args.learning_rate  # 当前学习率
                        logger.info(f"Step {completed_steps}: loss = {avg_loss:.4f}, lr = {current_lr:.2e}")
                        total_loss = 0  # 重置累计loss

                    if training_args.save_steps > 0 and completed_steps % training_args.save_steps == 0:  # 保存检查点
                        output_dir = os.path.join(training_args.output_dir, f"checkpoint-{completed_steps}")  # 组合输出路径
                        if model_is_distributed:
                            os.makedirs(output_dir, exist_ok=True)  # 创建目录以存放分布式权重
                            model.save_pretrained(output_dir)  # 保存LoRA/全参数权重
                            tokenizer.save_pretrained(output_dir)
                            torch.save({  # 保存优化器及LR调度器状态，便于恢复训练
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                                'completed_steps': completed_steps,
                            }, os.path.join(output_dir, 'training_state.pt'))
                        else:
                            accelerator.save_state(output_dir)  # Accelerate自动保存模型+优化器+随机状态
                        logger.info(f"保存检查点到: {output_dir}")

                    if (training_args.do_eval and
                            training_args.eval_steps > 0 and
                            completed_steps % training_args.eval_steps == 0 and
                            eval_dataloader is not None):  # 周期性评估
                        model.eval()  # 切换到评估模式
                        eval_loss = 0  # 重置评估loss累计
                        eval_steps = 0  # 重置评估步数

                        for eval_batch in eval_dataloader:  # 遍历评估批次
                            with torch.no_grad():  # 推理阶段关闭梯度
                                eval_outputs = model(**eval_batch)  # 前向推理得到loss
                                eval_loss += eval_outputs.loss.detach().float()  # 累加loss
                                eval_steps += 1  # 增加评估步计数

                        avg_eval_loss = eval_loss / eval_steps  # 计算平均验证损失
                        try:
                            perplexity = math.exp(avg_eval_loss)  # 困惑度=exp(平均loss)
                        except OverflowError:
                            perplexity = float("inf")  # Loss过大导致exp溢出时返回无穷

                        logger.info(
                            f"Step {completed_steps}: eval_loss = {avg_eval_loss:.4f}, perplexity = {perplexity:.2f}")
                        model.train()  # 评估后切回训练模式
        progress_bar.close()  # 结束进度条

        if accelerator.is_main_process:
            logger.info(f"保存模型到: {training_args.output_dir}")  # 仅主进程打印保存信息

        unwrapped = get_unwrapped_model(model)  # 提取原始模型对象
        unwrapped.config.use_cache = True  # 恢复use_cache设置，便于推理
        unwrapped.enable_input_require_grads()  # 恢复输入梯度需求（对推理无害）

        if model_is_distributed:
            logger.info("🔧 保存分布式模型...")  # 记录即将保存已经分片的模型
            model.save_pretrained(training_args.output_dir)  # 保存分布式权重（含LoRA）
            tokenizer.save_pretrained(training_args.output_dir)  # 保存分词器
        else:
            accelerator.wait_for_everyone()  # 同步所有进程

            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)  # 去除Accelerate包装
                save_model(unwrapped_model, tokenizer, training_args.output_dir)  # 调用统一保存函数
                logger.info("✅ 标准模型保存完成")

    if training_args.do_eval and eval_dataloader is not None:  # 终局评估
        logger.info("*** 最终评估 ***")  # 日志提示进入最终评估阶段
        model.eval()  # 切换评估模式
        eval_loss = 0  # 初始化loss累计
        eval_steps = 0  # 初始化步计数

        for eval_batch in eval_dataloader:  # 遍历全部验证批次
            with torch.no_grad():  # 禁止梯度计算
                eval_outputs = model(**eval_batch)  # 执行前向推理
                eval_loss += eval_outputs.loss.detach().float()  # 累加loss
                eval_steps += 1  # 增加评估步计数

        avg_eval_loss = eval_loss / eval_steps  # 求平均验证损失
        try:
            perplexity = math.exp(avg_eval_loss)  # 计算最终困惑度
        except OverflowError:
            perplexity = float("inf")  # 避免指数溢出
        if accelerator.is_main_process:
            logger.info(f"最终评估结果: eval_loss = {avg_eval_loss:.4f}, perplexity = {perplexity:.2f}")  # 主进程输出最终指标


if __name__ == "__main__":  # 当脚本直接执行时
    main()  # 运行主流程
