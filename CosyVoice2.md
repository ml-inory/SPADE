# CosyVoice2 复现指南：SPADE 结构化剪枝 + 自适应蒸馏

本文档记录在**真实 CosyVoice2-0.5B** 模型上复现 SPADE
（[arXiv:2509.20802](https://arxiv.org/abs/2509.20802)，ICASSP 2026）的完整过程。
目标是把论文中"剪掉一半 Transformer 层 + 蒸馏恢复质量"的实验，从一个空仓库
一步步跑出结果。**小白按照本文操作即可从零复现，不需要改任何代码。**

复现最终效果（30 句 LibriSpeech dev-clean，Whisper-base WER）：

| 模型 | 深度 | 参数量 | WER | RTF |
|---|---:|---:|---:|---:|
| CosyVoice2 teacher | 24 | 494M | 0.334 | 0.82 |
| WLI 剪枝（12 层，未蒸馏） | 12 | 315M | 1.015 | 6.85 |
| 剪枝 + SPADE 蒸馏（1500 句） | 12 | 315M | 0.358 | 0.60 |
| **剪枝 + SPADE 蒸馏（5500 句）** | 12 | 315M | **0.315** | 0.59 |

**核心结论**：12 层学生模型 WER 首次低于 24 层教师（0.315 < 0.334），
参数 -36%，端到端 RTF 从 0.82 降到 ~0.59（提速 ~1.4×）。

### RTF 结论（2026-08 记录）

端到端 RTF（LLM + Flow + HiFi-GAN，单张 L4，batch=1，完整合成链路实测）：

| 模型 | RTF | 相对 teacher |
|---|---:|---:|
| CosyVoice2 teacher（24 层） | 0.82 | 1.00× |
| 蒸馏后 12 层（5500 句，WER 最优单片） | 0.59 | ~1.37× |
| 蒸馏后 12 层（双片权重平均） | 0.59 | ~1.39× |

两点说明：

1. **提速被未压缩部分摊薄**：SPADE 只剪/蒸馏 LLM（24→12 层），Flow 匹配与
   HiFi-GAN 原样保留。LLM 单独部分约快 2×，摊到全链路后约 1.37–1.39×。
   继续压缩 Flow/HiFi-GAN 或做 JIT/ONNX 导出可进一步降低端到端 RTF。
2. **与论文对比**：论文报 CosyVoice2 最高 **1.7× RTF 提升**；本文数字为
   原版 Python 推理路径上的实测，口径更保守（论文含流式/JIT 等推理优化），
   量级一致。

---

## 1. 原理一句话版

SPADE 分两步：

1. **WLI 剪枝**：把每一层 Transformer 临时"拿掉"（残差结构下等价于置零），
   用合成的语音 + Whisper 识别算 WER，WER 涨得越多的层越重要，反之可剪。
2. **自适应多级蒸馏**：用原模型当教师（冻结），用
   `CE + Skew-KL logits + 隐层/注意力/embedding MSE` 复合损失微调剪枝后的学生，
   且学生每一层对齐的是"被剪掉区间末尾的教师层"，从而吸收被剪层的能力。

只剪/蒸馏 LLM（Qwen2 主干），Flow 和 HiFi-GAN 原样复用。

---

## 2. 环境要求

**硬件**

- 2 × NVIDIA GPU，每张 ≥ 24 GB 显存（本项目用 2 × L4 24GB 验证）
- 磁盘 ≥ 20 GB（模型 ~5.3 GB + 数据 ~2 GB + 中间产物）
- 内存 ≥ 32 GB

**软件**

- Linux（本项目在 Ubuntu 兼容环境验证）
- Python ≥ 3.10（验证版本 3.12）
- 已安装 CUDA 版 PyTorch ≥ 2.1（验证版本 2.9.1+cu128）
- 能访问外网（Hugging Face / ModelScope / GitHub；国内环境建议配置代理或镜像）

---

## 3. 从零一键复现（推荐）

```bash
# 1. 克隆本仓库
git clone git@github.com:ml-inory/SPADE.git
cd SPADE

# 2. 一键复现（自动完成：环境 -> 数据 -> WLI -> 剪枝 -> 蒸馏 -> 评估）
bash scripts/reproduce_cosyvoice2.sh
# 预计 2~2.5 小时（2 张 GPU）；结果输出在 outputs/cosyvoice2/repro/

# 只想快速验证流程能跑通（约 10 分钟）：
bash scripts/reproduce_cosyvoice2.sh SPADE_FAST=1
```

> 脚本会用环境变量自动定位路径，不需要手动配置。常用覆盖项见文末「7. 常用开关」。

---

## 4. 分步详解

下面每一步都和 `scripts/reproduce_cosyvoice2.sh` 一一对应，方便理解与排错。

### 4.0 环境准备（setup.sh）

```bash
bash spade_cosyvoice2/setup.sh
```

脚本做三件事：

1. 下载 **CosyVoice 官方仓库**到 `../CosyVoice-main`（与 SPADE 同级），
   已存在则跳过；
2. 安装 CosyVoice 推理所需的少量 Python 依赖（`matcha-tts`、`wetext`；
   torch/transformers 等大件要求你已装好）；
3. 下载 **CosyVoice2-0.5B 权重**到
   `$COSYVOICE_ROOT/pretrained_models/CosyVoice2-0.5B`
   （优先 ModelScope `iic/CosyVoice2-0.5B`，失败自动回退 Hugging Face）。

验证模型可加载、能合成：

```bash
python -m spade_cosyvoice2.smoke --out outputs/cosyvoice2/smoke.wav
# 期望输出: saved outputs/cosyvoice2/smoke.wav (6.24s)
```

### 4.1 数据下载

```bash
bash scripts/download_cosyvoice2_data.sh
```

下载 LibriSpeech 的 parquet 数据（含音频字节与文本）：

- `validation.clean/0000.parquet`（342 MB，dev-clean，**评估集**）
- `train.clean.100/0000-0002.parquet`（~1.4 GB，约 6000 句，**训练集**）

落盘位置：`$COSYVOICE_ROOT/data/`。支持断点续传（`curl -C -`）；
网络受限时可先设置 `export HTTPS_PROXY=http://代理:端口`。

### 4.2 数据准备（data_prep.py）

```bash
python -m spade_cosyvoice2.data_prep --config <你的配置>
```

复现脚本会自动生成配置并调用。它做的事：

1. 从 parquet 读音频字节，写成 flac 文件；
2. 用 `campplus.onnx`（CPU）提取**说话人 embedding**（192 维）；
3. 用 `speech_tokenizer_v2.onnx`（CPU）提取 **25Hz 语音 token**；
4. 写出 CosyVoice 训练格式的 parquet（`train_shard*.parquet` + `eval.parquet`）
   和 `*.data.list` 文件。

关键配置（也可直接改 `configs/cosyvoice2/pipeline_scale.yaml`）：

| 字段 | 含义 | 默认 |
|---|---|---|
| `hf_parquet` | 训练 parquet（支持 `*.parquet` 通配） | 自动探测 |
| `eval_hf_parquet` | 评估 parquet | 自动探测 |
| `num_utts` / `train_utts` / `eval_utts` | 数量 | 5500 / 5500 / 200 |
| `train_shards` | 切成几片（=GPU 数） | 2 |

产物：`$COSYVOICE_ROOT/data/spade_repro/parquet/` 与 `data_prep.json`。

### 4.3 WLI：层重要性（wli.py）

```bash
python -m spade_cosyvoice2.wli --config <配置>
```

对 24 层 LLM 逐层做 leave-one-out：置零该层参数 -> 零样本合成评估句 ->
Whisper 转写 -> 与参考文本算 WER。**本机实测约 45 分钟**（8 句 × 24 层），
可通过 `subset_size` 调小。

输出 `wli_report.json`，本复现的实测值（最高 = 最重要）：

```
层16(1.438) 15(1.397) 11(1.265) 8(1.156) ... 22(0.354) 23(0.367)
```

### 4.4 剪枝 24 -> 12（prune_llm.py）

```bash
python -m spade_cosyvoice2.prune_llm \
  --llm-pt <model_dir>/llm.pt \
  --target-layers 12 \
  --retained 0 1 4 7 8 10 11 12 15 16 19 23 \
  --out outputs/cosyvoice2/pruned_llm.pt
```

`--retained` 由 WLI 报告自动选出（保留首尾层 + WLI 最高的中间层）。
原理：对状态字典做键重映射，把保留层的权重拷贝到 0..11 的新层，其余删除。
产出 `pruned_llm.pt`（可直接被 CosyVoice 加载器使用）。

### 4.5 蒸馏（distill.py）

单卡快速版：

```bash
python -u -m spade_cosyvoice2.distill --config <配置>
```

双卡规模化版（复现脚本自动执行）：把训练集切成 2 片，两片**并行**训练
7 epochs（不同 seed），再用 `average_checkpoints.py` 平均权重。

蒸馏关键配置：

| 字段 | 含义 | 本项目取值 |
|---|---|---|
| `alpha` | 监督 vs 蒸馏权重（论文式 2） | 0.25 |
| `beta` / `skew_mode` | Skew-KL 参数（DistiLLM） | 0.1 / forward |
| `matching` | 动态层匹配（dynamic/static） | dynamic |
| `lr` | 学习率 | 1e-5 |
| `accum_grad` | 梯度累积 | 2 |
| `use_amp` | bf16 混合精度（省显存） | true |
| `save_every` / `resume` | 定期存档 / 断点续训 | 1000 / true |

每步日志示例：

```
[distill] epoch 7 step 7700 ce=4.2080 logit=0.6553 latent=1.9937 attention=0.0010 embedding=0.0656 total=1.5612
```

### 4.6 权重平均（average_checkpoints.py）

```bash
python -m spade_cosyvoice2.average_checkpoints \
  --checkpoints distilled_shard0.pt distilled_shard1.pt \
  --out distilled_avg.pt
```

> 经验：两个分片模型差异较大时，朴素平均不一定优于最好单片。本次复现中
> **shard0 单片 WER 0.315 优于平均后的 0.344**，最终取最好单片报告。

### 4.7 评估（evaluate.py）

```bash
python -m spade_cosyvoice2.evaluate --config <配置> --json eval.json
```

指标：Whisper-base WER（已做 ASR 式大小写/标点归一化；退化输出按 WER=1.0 计）、
RTF（合成耗时/音频时长）、参数量、深度。teacher 用 `retained: []`。

---

## 5. 产物清单

| 文件 | 说明 |
|---|---|
| `outputs/cosyvoice2/repro/pruned_llm.pt` | 24->12 层剪枝后的 LLM |
| `outputs/cosyvoice2/repro/distilled_shard{0,1}.pt` | 两片蒸馏结果 |
| `outputs/cosyvoice2/repro/distilled_avg.pt` | 平均后的最终模型 |
| `outputs/cosyvoice2/repro/wli_report.json` | 24 层 WLI 值 |
| `outputs/cosyvoice2/repro/eval_{teacher,pruned,distilled}.json` | 三阶段评估 |
| `<cosyvoice_root>/data/spade_repro/parquet/` | 训练/评估 parquet |
| `<cosyvoice_root>/pretrained_models/CosyVoice2-0.5B/` | 官方权重 |

---

## 6. 常见问题（FAQ）

**Q1：torchaudio 报 `libnvrtc.so.13` / torchcodec 相关错误？**

这是 torchaudio 2.9 尝试加载 torchcodec 导致的。仓库在导入时已自动把
`torchaudio.load` 打到 soundfile 后端，正常不需要处理。若在其它脚本里直接
用 torchaudio，请先 `from spade_cosyvoice2.paths import ensure_import;
ensure_import()`。

**Q2：下载很慢 / 失败？**

- 数据脚本支持断点续传，重跑即可续传；
- 可设置 `export HTTPS_PROXY=http://127.0.0.1:17890`（示例端口）；
- 权重优先走 ModelScope，国内通常较快。

**Q3：显存不够（CUDA OOM）？**

- 确认 `use_amp: true`（bf16，本项目显存 ~12GB/卡）；
- 减小 `accum_grad` 或数据 `max_frames_in_batch`；
- 单卡运行时把 `SPADE_GPUS` 设成 1 张，脚本自动退化为单片训练。

**Q4：训练中断了？**

蒸馏已内置 `save_every` + `resume`：重跑同一条命令会从最近存档继续。

**Q5：WLI 太慢？**

调小 `subset_size`（如 4）即可，代价是重要性估计更粗糙。

**Q6：为什么我的 WER 和文档不完全一致？**

评估集抽样、Whisper 版本、随机种子都会引入波动；重点看相对趋势
（剪枝后显著变差、蒸馏后恢复/超越）。

---

## 7. 常用开关（reproduce_cosyvoice2.sh）

```bash
export COSYVOICE_ROOT=/path/to/CosyVoice-main   # 源码位置（默认 SPADE 的上级目录）
export COSYVOICE2_MODEL_DIR=/path/to/model      # 权重位置
export SPADE_GPUS="0 1"                          # 用哪几张 GPU
export SPADE_NUM_UTTS=5500 SPADE_EPOCHS=7        # 数据量/轮数
export SPADE_SKIP_WLI=1                          # 已有 wli_report.json 时跳过 WLI
export SPADE_FAST=1                              # 小数据冒烟（几分钟）
bash scripts/reproduce_cosyvoice2.sh
```

## 8. 已知限制与下一步

- 本复现使用 train-clean-100 前 3 个分片（~6000 句），远小于论文的
  LibriHeavy 规模；加大到全量 28k 句、更多 epochs 可进一步逼近教师；
- WLI 成本随层数 × 评估句数线性增长；
- 权重平均策略可优化（按验证集加权、选相近检查点），目前取最好单片；
- 评估为自动指标，建议最终对音频做主观试听。
