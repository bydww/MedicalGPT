# -*- coding: utf-8 -*-  # 指定源码编码，防止中文注释在不同平台下出现乱码
"""  # 模块级文档字符串：说明脚本作者与用途
author:XuMing(xuming624@qq.com)
description: Train a model from SFT using PPO
"""

import os  # 导入os以便读取环境变量、执行路径判断
from dataclasses import dataclass, field  # dataclass用于快速定义参数结构体，field指定默认值
from glob import glob  # glob用于递归匹配文件路径列表
from typing import Optional  # Optional用于类型提示，标记参数可为None
from datasets import load_dataset  # Hugging Face datasets加载API，返回Dataset或DatasetDict
from loguru import logger  # loguru提供丰富的日志接口，便于跟踪训练流程
from transformers import (  # 从transformers加载所需模型与工具类
    AutoModelForSequenceClassification,  # 自动实例化序列分类模型，本文用于奖励/价值模型
    AutoTokenizer,  # 自动加载对应的分词器
    HfArgumentParser,  # 将命令行参数解析为dataclass对象
    AutoModelForCausalLM,  # 自动实例化因果语言模型，作为PPO策略模型
)
from trl import (  # trl库提供RLHF/PPO训练工具
    PPOConfig,  # PPO训练超参数配置
    PPOTrainer,  # PPO训练主控类，封装采样、优势计算、策略更新
    ModelConfig,  # 模型相关配置（如LoRA设置）
    get_peft_config,  # 根据ModelConfig生成PEFT配置（如LoRA/QLoRA）
)
from template import get_conv_template  # 自定义模板函数，将对话JSON转为模型提示词字符串

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"  # 禁用tokenizers多线程，避免tokenizer警告或竞争
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许MKL重复加载，防止在某些环境下崩溃


@dataclass  # 使用dataclass定义PPOArguments以便解析命令行参数
class PPOArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """  # 原始英文文档说明：PPO微调所需的基础因果语言模型
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})  # 数据集名称；None表示使用本地文件
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})  # 数据集子配置
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})  # 指定训练集划分名
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})  # 指定评估集划分名（若存在）
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})  # 本地训练数据目录
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )  # 本地验证数据目录
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})  # 对话模板名称，决定prompt格式
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})  # 限制prompt最大token数，防止超长


def main():  # 主函数，完成参数解析、模型加载、数据预处理和训练流程
    parser = HfArgumentParser((PPOArguments, PPOConfig, ModelConfig))  # 创建解析器，将命令行映射到三个dataclass对象
    args, training_args, model_args = parser.parse_args_into_dataclasses()  # 解析命令行，返回PPOArguments、PPOConfig、ModelConfig实例

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # 读取分布式环境变量LOCAL_RANK，默认为0；用于区分进程
    is_main_process = local_rank == 0  # 判断当前进程是否为主进程（rank=0），用于控制日志与保存

    if is_main_process:  # 仅主进程打印解析到的参数，避免多卡重复输出
        logger.info(f"Parse args: {args}")  # 输出PPOArguments内容
        logger.info(f"Training args: {training_args}")  # 输出PPOConfig内容（含学习率、batch等）
        logger.info(f"Model args: {model_args}")  # 输出ModelConfig内容（如LoRA设置）

    tokenizer = AutoTokenizer.from_pretrained(  # 加载SFT模型对应的分词器，保持与策略模型一致
        training_args.sft_model_path,  # 路径指向已完成监督微调的模型，用于奖励模型对齐
        trust_remote_code=model_args.trust_remote_code  # 是否允许执行远程仓库自定义代码
    )  # 结束from_pretrained参数列表
    if tokenizer.eos_token_id is None:  # 如果分词器缺少EOS标记，需要补全
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token  # 优先复用已有的eos/sep
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})  # 向词表添加EOS，保证decode时有终止符
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")  # 记录新增的token及其ID
    if tokenizer.bos_token_id is None:  # 如果缺少BOS，同样补齐
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})  # 直接使用EOS的符号作为BOS
        tokenizer.bos_token_id = tokenizer.eos_token_id  # 对齐ID，保证encode时首token存在
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")  # 打印添加情况
    if tokenizer.pad_token_id is None:  # 如果缺少PAD，需要设置以便批量padding
        if tokenizer.unk_token_id is not None:  # 优先使用UNK作为PAD，避免重复符号影响样本
            tokenizer.pad_token = tokenizer.unk_token  # 设置pad_token指向unk符号
        else:
            tokenizer.pad_token = tokenizer.eos_token  # 若无UNK则退而求其次使用EOS
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")  # 记录pad_token实际指向
    logger.debug(f"Tokenizer: {tokenizer}")  # 调试日志展示分词器对象，方便确认特殊符号

    value_model = AutoModelForSequenceClassification.from_pretrained(  # 加载价值模型，输出形状(batch,1)
        training_args.reward_model_path,  # reward_model_path指向已训练的RM权重
        trust_remote_code=model_args.trust_remote_code,  # 允许远程自定义代码
        num_labels=1  # 设置分类头输出1维标量，用于估计V(s)
    )  # 结束from_pretrained参数列表
    reward_model = AutoModelForSequenceClassification.from_pretrained(  # 加载奖励模型，提供即时reward估计
        training_args.reward_model_path,  # 同一权重路径，确保奖励与价值初始化一致
        trust_remote_code=model_args.trust_remote_code,  # 保持一致的信任策略
        num_labels=1  # 输出shape同样为(batch,1)，表示奖励得分
    )  # 结束from_pretrained参数列表
    policy = AutoModelForCausalLM.from_pretrained(  # 加载PPO策略模型，初始为SFT权重
        training_args.sft_model_path,  # 使用SFT模型作为PPO起点
        trust_remote_code=model_args.trust_remote_code  # 允许自定义模型结构
    )  # 结束from_pretrained参数列表

    peft_config = get_peft_config(model_args)  # 根据model_args（如是否LoRA）生成PEFT配置；若未启用返回None
    if peft_config is None:  # 当不使用PEFT时需要显式加载参考策略
        ref_policy = AutoModelForCausalLM.from_pretrained(  # 参考策略固定为初始SFT模型，用于计算KL散度
            training_args.sft_model_path,  # 同source路径，保证初始参数一致
            trust_remote_code=model_args.trust_remote_code  # 与策略模型保持一致的信任设置
        )  # 结束from_pretrained参数列表
    else:
        ref_policy = None  # 若使用LoRA，PPOTrainer会内部共享策略权重，无需独立参考模型

    prompt_template = get_conv_template(args.template_name)  # 获取对话模板，将ShareGPT格式转成prompt字符串
    if args.dataset_name is not None:  # 若提供远程数据集名称，则从HF Hub加载
        dataset = load_dataset(  # 加载指定数据集split，返回Dataset对象
            args.dataset_name,  # 数据集仓库名
            args.dataset_config,  # 子配置，如语言或子任务
            split=args.dataset_train_split  # 选择训练划分，例如"train"
        )  # 结束load_dataset参数列表
        eval_samples = 100  # 固定从训练集中切100条作为评估集
        train_dataset = dataset.select(range(len(dataset) - eval_samples))  # 切片选择前N-100条作为训练集
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))  # 取最后100条作为评估集
    else:
        data_files = {}  # 初始化文件映射字典，键为split名，值为文件路径列表
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):  # 若指定训练目录且存在
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(  # 搜索所有json文件
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)  # 搜索所有jsonl文件
            logger.info(f"train files: {', '.join(train_data_files)}")  # 打印找到的训练文件列表
            data_files["train"] = train_data_files  # 登记到data_files，供load_dataset使用
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):  # 若指定验证目录
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(  # 搜索验证json文件
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)  # 搜索验证jsonl文件
            logger.info(f"eval files: {', '.join(eval_data_files)}")  # 打印验证文件列表
            data_files["validation"] = eval_data_files  # 添加到data_files
        dataset = load_dataset(  # 使用json加载器读取本地文件，返回DatasetDict
            'json',  # 指定数据格式解析器
            data_files=data_files,  # 传入分split的文件映射
        )  # 结束load_dataset参数列表
        train_dataset = dataset["train"]  # 获取训练split，类型为Dataset
        val_dataset = dataset["validation"]  # 获取验证split
        eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))  # 取最多100条验证样本作为评估集
    logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")  # 打印训练/评估数据集信息（包含样本数、列名）

    max_source_length = args.max_source_length  # 记录prompt最大长度限制（当前流程未裁剪，可根据需要扩展）

    def preprocess_function(examples):  # 定义datasets.map使用的预处理函数，输入为一个批量examples
        new_examples = {"input_ids": []}  # 初始化输出字典；将收集每条样本的prompt token序列（List[List[int]]）
        roles = ["human", "gpt"]  # 预期对话角色顺序：human提问、gpt回答

        def get_dialog(examples):  # 内部生成器，将ShareGPT格式转换为模板化的对话列表
            system_prompts = examples.get("system_prompt", "")  # 读取批量中的system提示列表，若不存在返回空串
            for i, source in enumerate(examples['conversations']):  # 遍历batch中每条conversation（List[dict]）
                if len(source) < 2:  # 若对话长度不足一轮问答，无法用于训练
                    continue  # 跳过该样本
                data_role = source[0].get("from", "")  # 读取第一条消息角色
                if data_role not in roles or data_role != roles[0]:  # 如果第一条不是human，说明前缀含系统或异常，需要跳过第一条
                    source = source[1:]  # 删除首条消息，保证剩余序列以human开始
                if len(source) < 2:  # 调整后若仍不足一轮问答
                    continue  # 跳过
                messages = []  # 用于收集按顺序的文本内容
                for j, sentence in enumerate(source):  # 遍历消息序列
                    data_role = sentence.get("from", "")  # 当前消息角色
                    if data_role not in roles:  # 若角色既非human也非gpt，视为脏数据
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")  # 记录警告并跳出
                        break
                    if data_role == roles[j % 2]:  # 检查角色是否按human/gpt交替
                        messages.append(sentence["value"])  # 收集文本内容
                if len(messages) < 2 or len(messages) % 2 != 0:  # 如果无法形成完整的问答配对
                    continue  # 跳过样本
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]  # 组合成[[问,答], ...]
                system_prompt = system_prompts[i] if system_prompts else None  # 若batch提供system_prompt则取对应项
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)  # 使用模板生成扁平化字符串列表 [prompt0, answer0, ...]

        for dialog in get_dialog(examples):  # 遍历生成的dialog列表（长度=2*n轮）
            for i in range(len(dialog) // 2):  # 只保留human侧文本用于生成问题，故按轮次取偶数索引
                source_txt = dialog[2 * i]  # 获取第i轮的用户prompt字符串
                tokenized_question = tokenizer(source_txt, padding=False)  # 使用分词器编码，不做padding；输出dict含input_ids(List[int])
                new_examples["input_ids"].append(tokenized_question["input_ids"])  # 将编码后的token序列追加到输出；形状追加后为List[num_samples][prompt_len]

        return new_examples  # 返回字典供datasets.map合并；缺失的字段自动被忽略

    if is_main_process:  # 仅主进程执行预处理，避免多卡重复写缓存；其余进程会等待广播
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")  # 输出原始训练样本供检查（包含conversations字段）
        tokenized_train_dataset = train_dataset.map(  # 对训练数据应用预处理函数，生成token序列
            preprocess_function,  # 指定映射函数
            batched=True,  # 启用批处理，examples为dict[str, List[Any]]
            num_proc=training_args.dataset_num_proc,  # 多进程加速映射；值通常为CPU核数
            remove_columns=train_dataset.column_names,  # 移除原有字段，仅保留preprocess返回的input_ids
            load_from_cache_file=False,  # 每次重新处理，避免旧缓存影响
            desc="Running tokenizer on dataset" if is_main_process else None,  # 设置进度条描述
        )  # 结束map调用参数列表
        train_dataset = tokenized_train_dataset.filter(  # 过滤掉input_ids为空的样本
            lambda x: len(x['input_ids']) > 0  # 保留至少包含一个token的prompt
        )  # 结束filter调用，返回新的Dataset实例
        logger.debug(f"Train samples tokenized top3: {train_dataset[:3]}")  # 打印前3条token化后的训练样例

        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")  # 显示原始评估样本结构
        tokenized_eval_dataset = eval_dataset.map(  # 对评估数据执行同样的token化流程
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )  # 结束map调用参数列表
        eval_dataset = tokenized_eval_dataset.filter(  # 过滤掉空prompt
            lambda x: len(x['input_ids']) > 0
        )  # 结束filter调用
        logger.debug(f"Eval samples tokenized top3: {eval_dataset[:3]}")  # 输出token化后的评估样本（前3条）

    trainer = PPOTrainer(  # 构建PPOTrainer，将策略/参考/奖励/价值模型以及数据集注入
        args=training_args,  # PPOConfig，包含学习率、batch_size、KL系数等
        processing_class=tokenizer,  # 指定tokenizer用于文本->token转换与解码
        model=policy,  # 需优化的策略模型π_θ
        ref_model=ref_policy,  # 参考策略π_φ，用于KL惩罚；可能为None
        reward_model=reward_model,  # 奖励模型R_ψ，输入(prompt+response)输出标量奖励
        value_model=value_model,  # 价值模型V_ψ，用于基线估计与优势计算
        train_dataset=train_dataset,  # 训练数据集；元素为{"input_ids": List[int]}
        eval_dataset=eval_dataset,  # 评估数据集；同上
        peft_config=peft_config,  # LoRA/QLoRA配置，若None则全量更新
    )  # 结束PPOTrainer初始化参数列表

    if training_args.do_train:  # 当PPOConfig.do_train=True时执行训练阶段
        if is_main_process:  # 仅主进程打印训练开始的提示
            logger.info("*** Train ***")  # 记录开始训练
        trainer.train()  # 调用PPOTrainer.train()：循环采样回应、计算奖励、更新策略；内部数据流
        # prompt ids (batch, prompt_len) -> 采样生成response ids (batch, response_len)
        # -> 拼接文本送入reward_model得到reward(batch,1)
        # -> 计算logits、KL、优势A_t并更新策略与value

        if is_main_process:  # 训练完成后仅主进程保存策略模型
            trainer.save_model(training_args.output_dir)  # 将策略模型权重保存到输出目录；若使用LoRA仅保存适配器

    trainer.generate_completions()  # 调用生成接口，使用最终策略模型对eval_dataset生成样本，用于人工评估或可视化


if __name__ == "__main__":  # 当脚本作为主程序执行时
    main()  # 调用main()启动整个PPO流程
