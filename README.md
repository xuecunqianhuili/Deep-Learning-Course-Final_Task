# 中文新闻自动摘要与关键字生成系统 (深度学习大作业)

本项目基于 **mT5-small** 模型实现中文文本自动摘要，并结合 **KeyBERT** 实现无监督关键词提取。项目针对 **NVIDIA RTX 5070** 等新一代显卡进行了 **BF16** 训练优化，解决了 T5 模型在 FP16 下的梯度溢出问题。

## 1. 目录结构
- `data/`: 数据加载与预处理逻辑
- `model/`: 模型定义 (mT5 Wrapper, KeyBERT)
- `train/`: 训练脚本 (基于 HuggingFace Trainer)
- `utils/`: 工具函数 (ROUGE 指标计算, 可视化绘图)
- `demo/`: 演示脚本 (整合摘要与关键字生成)
- `results/`: 存放实验结果 (模型权重, 训练曲线, 评估指标, Demo输出)

## 2. 环境搭建 (Windows + RTX 50 Series)

由于 RTX 5070 架构较新 (Blackwell/sm_120)，需要使用 PyTorch Nightly 版本配合 CUDA 12.8 以获得最佳支持。

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活环境 (PowerShell)
& ".\.venv\Scripts\Activate.ps1"
# 或者如果策略报错，直接使用 python 绝对路径运行 (推荐)

# 3. 安装基础依赖
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# 4. 安装适配 RTX 5070 的 PyTorch (Nightly + CUDA 12.8)
& ".\.venv\Scripts\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

## 3. 运行指南

**注意**：以下命令建议使用虚拟环境中的 python 解释器路径运行，例如 `& "path/to/.venv/Scripts/python.exe" script.py`。

### 3.1 训练模型 (Train)
默认使用 `CSEBUETNLP/xlsum` (中文部分) 数据集。
针对 RTX 5070 已自动开启 **BF16 mixed precision** 训练。

```powershell
# 完整训练 (推荐设置: 3-5 epochs, batch size 8)
& ".\.venv\Scripts\python.exe" train/train_summarization.py --epochs 3 --batch_size 8 --output_dir results/summarizer_mt5_small

# 快速测试 (仅调试用)
& ".\.venv\Scripts\python.exe" train/train_summarization.py --subset_size 100 --epochs 1
```

训练完成后，`results/summarizer_mt5_small` 目录下会生成：
- `training_curves.png`: Loss 和 ROUGE 变化曲线
- `data_distribution.png`: 数据集字数分布图
- `model.safetensors`: 训练好的模型权重

### 3.2 评估模型 (Evaluate)
计算测试集上的 ROUGE-1, ROUGE-2, ROUGE-L 分数（包含中文分词优化）。

```powershell
& ".\.venv\Scripts\python.exe" evaluate_final.py --subset 200
```

### 3.3 演示功能 (Demo)
输入一段长文本，输出摘要和关键词。

```powershell
& ".\.venv\Scripts\python.exe" demo/demo.py --text "在此处输入需要摘要的一段长新闻文本..."
```

## 4. 技术细节与创新点
- **模型选型**: Google mT5 (Multilingual T5)，支持多语言 seq2seq 任务。
- **硬件优化**: 针对 Blackwell 架构 (RTX 5070) 适配 CUDA 12.8。
- **精度策略**: 采用 **BF16 (BFloat16)** 替代传统的 FP16，解决了 T5 模型在半精度训练时的数值不稳定性 (NaN Loss)，同时保持了训练速度。
- **中文处理**: 引入 `jieba` 对 ROUGE 评估进行中文分词预处理，使指标更符合中文语境。

## 5. 项目引用
- Model: `google/mt5-small`
- Dataset: `csebuetnlp/xlsum`
- Keyword Generation: `KeyBERT` (`sentence-transformers`)
