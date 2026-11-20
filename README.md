# 单层 Transformer 修复头全词表 FKL

独立运行已登记的 `TR1-C-FKL-I0`。冻结原训练源码、初始权重绑定、5 个训练轮次和开发组选权重协议。损失用完整 Target 分布的 FKL，草稿第 1 至第 8 位按 8 至 1 加权。训练结束后只评校准交叉熵最低的一套权重。最终对照是同一 503 题上的 RAW，每轮接受的草稿 token 用总接受数除以总验证轮数计算。

## 第二台 A100

1. 使用 Linux、Python 3.11+ 和与 CUDA 驱动匹配的 PyTorch，再安装 `requirements.txt`。先运行 `python experiment.py check-repo` 核对 Git 中的原源码字节。
2. 把第一台机器的冻结输入按原 `/data` 目录层次复制或挂载到第二台。这个旧入口的路径写在冻结源码中。`native` 的 `runtime_freeze.json` 列出精确文件。权重、训练缓存、RAW 特征、题单和旧阶段完成回执从原机输入快照准备。实验运行和缓存输出所在目录需要可写。
3. 在第二台执行下面命令。`install` 逐个验证并补齐仓库中的源码。它遇到与锁定版本不同的已有文件会报错。两条旧 FKL 在同一 `/data` 下可复用同一套 `fkl/gate` 和 `fkl/backfill`。同一机器由原 GPU 文件锁串行受理不同实验。

```bash
set -e
python experiment.py check-repo
python experiment.py install
python experiment.py preflight --verify-hashes
python experiment.py gate
python experiment.py backfill
python experiment.py smoke
python experiment.py train
python experiment.py eval
```

原训练和评估有各自的断点。重复运行会按封存回执和权重哈希续接。训练输出在 `continuation-v5-native-logits-v1/runs/TR1-C-FKL-I0/train`，逐题评估在同级 `eval/secondary`。本仓库保留的是原机已登记的 v5 代码与单权重评估适配器。第二台机器的真机前向和完整评估需在输入迁移后验证。

## 原生教师 logits 的训练前验证

在本地运行 `python test_native_logits.py`，用 BF16 夹具验证完全一致、稀疏差异、密集差异和损坏分片。夹具覆盖 A100 已观察到的首个差异值和 0.0625 最大绝对误差。真机数值需在同一 CUDA 环境、真实 Qwen3-8B BF16 权重与冻结前缀上运行 `python experiment.py gate`。GPU 执行席沿用原机 GPU0 锁及已核实的单卡资源检查。

新运行使用 `continuation-v5-native-logits-v1`，两种 FKL 训练共享此版本的教师缓存。旧运行的阻塞记录保留在 `continuation-v5`。新 gate 校验原生贪心、RAW 的 Top32、logsumexp 与候选分数，再从磁盘用训练读取函数恢复整行，检查原生 FP32 字节 SHA-256。各前缀的差异数、最大绝对差、存储模式及原生摘要见 `fkl/gate/native_roundtrip.json`。该回执和同目录的完成记录共同用于 backfill 准入。

稀疏差异存索引与原生值，密集差异存整行，完全一致只存 hidden 和原生摘要。训练每次读取均核整行原生摘要。数值格式、读取器和写入器的版本均计入新的冻结文件。目标机 gate 失败会返回非零，完整训练和 503 题评估各自需要对应的真机回执。
