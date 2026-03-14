# CEVAE项目环境配置攻略

> **项目**: Causal Effect Variational Autoencoder (CEVAE)
> **论文**: Louizos et al., "Causal Effect Inference with Deep Latent-Variable Models", NeurIPS 2017
> **适用场景**: 因果推断、深度学习、变分推断

---

## 快速开始

```bash
# 1. 创建conda环境
conda create -n cevae python=3.5 -y

# 2. 安装依赖
conda activate cevae

# 3. 安装核心包
pip install tensorflow==1.1.0 protobuf==3.5.2
pip install edward==1.3.1 progressbar2==3.34.3

# 4. 安装科学计算包（通过conda，避免编译问题）
conda install -c conda-forge scikit-learn scipy numpy -y

# 5. 创建模型保存目录
mkdir models

# 6. 运行实验
python cevae_ihdp.py
```

---

## 详细配置步骤

### 第一步：创建Conda环境

**为什么需要Python 3.5？**

TensorFlow 1.1.0 只支持 Python 3.5，不支持更高版本。

```bash
# Windows
C:\Users\<username>\miniconda3\condabin\conda.bat create -n cevae python=3.5 -y

# Linux/Mac
conda create -n cevae python=3.5 -y
```

**常见问题**：
- 如果Python 3.5不可用，尝试 `python=3.5.6` 或 `python=3.6`
- Python 3.6 需要使用 TensorFlow 1.2+，可能与代码不兼容

---

### 第二步：激活环境

```bash
# Windows
conda activate cevae

# Linux/Mac
source activate cevae
```

---

### 第三步：安装TensorFlow 1.1.0

**关键依赖顺序**：必须先安装旧版protobuf，否则会报错！

```bash
# 方法1：先安装兼容的protobuf
pip install protobuf==3.5.2
pip install tensorflow==1.1.0

# 方法2：一行命令
pip install protobuf==3.5.2 tensorflow==1.1.0
```

**常见错误与解决方案**：

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `protobuf requires Python '>=3.7'` | pip自动安装新版protobuf | `pip install protobuf==3.5.2 tensorflow==1.1.0` |
| `no matching distribution found` | Python版本不对 | 使用Python 3.5 |

---

### 第四步：安装Edward

Edward是建立在TensorFlow之上的概率编程库。

```bash
pip install edward==1.3.1
```

---

### 第五步：安装进度条库

原项目使用progressbar 2.3，但该版本与Python 3.5不兼容。

**解决方案**：使用progressbar2（API兼容）

```bash
pip install progressbar2==3.34.3
```

**代码修改**（cevae_ihdp.py 第158-163行）：

```python
# 原代码:
pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
pbar.start()
...
pbar.update(j)

# 修改为:
pbar = ProgressBar(max_value=n_iter_per_epoch, widgets=widgets)
pbar.start()
...
pbar.update(j + 1)
```

---

### 第六步：安装Scikit-learn和Scipy

**重要**：使用conda安装，避免从源码编译scipy的问题。

```bash
conda install -c conda-forge scikit-learn scipy numpy -y
```

**为什么不使用pip？**

- scipy 0.19.x（与sklearn 0.18兼容）需要从源码编译
- 编译需要Fortran编译器（Windows上很难配置）
- conda提供预编译的二进制包

**版本说明**：
- 原README要求：scikit-learn==0.18.1
- conda-forge提供：scikit-learn==0.20.0
- 两者API兼容，可以使用更新版本

---

### 第七步：验证安装

```python
import tensorflow as tf
print(tf.__version__)  # 应输出: 1.1.0

import edward
print(edward.__version__)  # 应输出: 1.3.1

import sklearn
print(sklearn.__version__)  # 应输出: 0.20.x

import scipy
print(scipy.__version__)  # 应输出: 1.1.x

import numpy as np
print(np.__version__)  # 应输出: 1.15.x
```

---

## 完整依赖清单

### requirements.txt (供参考)

```txt
# 深度学习框架
tensorflow==1.1.0
edward==1.3.1

# 进度条（替代原progressbar 2.3）
progressbar2==3.34.3

# 科学计算（建议通过conda安装）
scikit-learn>=0.18.1
scipy>=1.0.0
numpy>=1.15.0
```

---

## 数据集准备

项目已包含IHDP数据集，位于 `datasets/IHDP/csv/`：

```
datasets/IHDP/csv/
├── ihdp_npci_1.csv
├── ihdp_npci_2.csv
├── ...
└── ihdp_npci_10.csv
```

**数据格式**（每行一个样本）：
- 列0: 处理变量 t
- 列1: 结果变量 y
- 列2: 反事实结果 y_cf
- 列3: 控制组潜在结果 μ0
- 列4: 处理组潜在结果 μ1
- 列5-29: 协变量 x (25维)

---

## 运行实验

### 基本运行

```bash
python cevae_ihdp.py
```

### 自定义参数

```bash
# 调整重复次数
python cevae_ihdp.py -reps 50

# 调整学习率
python cevae_ihdp.py -lr 0.0005

# 调整训练轮数
python cevae_ihdp.py -epochs 150

# 调整早停检查频率
python cevae_ihdp.py -earl 5

# 调整输出频率
python cevae_ihdp.py -print_every 5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-reps` | 10 | 重复实验次数 |
| `-epochs` | 100 | 每次实验的训练轮数 |
| `-lr` | 0.001 | 学习率 |
| `-earl` | 10 | 早停检查频率（epoch数） |
| `-print_every` | 10 | 输出训练统计的频率 |
| `-opt` | adam | 优化器（adam/adamax） |

---

## 常见问题排查

### 问题1: ImportError: No module named 'scipy'

**原因**：scipy未安装

**解决**：
```bash
conda install -c conda-forge scipy -y
```

---

### 问题2: ImportError: cannot import name 'comb'

**原因**：scipy和scikit-learn版本不兼容

**解决**：
```bash
conda install -c conda-forge scikit-learn scipy -y
```

---

### 问题3: ValueError: progressbar API错误

**原因**：使用了原progressbar 2.3，与Python 3.5不兼容

**解决**：
```bash
pip install progressbar2==3.34.3
```

并修改代码（见第五步）

---

### 问题4: TensorFlow warnings (SSE/AVX instructions)

**现象**：
```
W tensorflow\core\platform\cpu_feature_guard.cc:45]
The TensorFlow library wasn't compiled to use SSE instructions...
```

**原因**：TensorFlow 1.1.0预编译包未启用CPU优化指令

**影响**：仅影响运行速度，不影响结果

**解决**：忽略警告，或从源码编译TensorFlow（不推荐）

---

### 问题5: CUDA/GPU相关错误

**现象**：
```
cudart64_80.dll not found
```

**原因**：TensorFlow 1.1.0 默认需要CUDA 8.0

**解决**：
```python
# 在代码开头添加
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU
```

---

## 项目结构

```
CEVAE/
├── cevae_ihdp.py      # 主程序：CEVAE模型训练
├── datasets.py        # 数据集加载（IHDP类）
├── evaluation.py      # 评估指标计算
├── utils.py           # 工具函数（神经网络构建等）
├── models/            # 模型保存目录（运行时创建）
├── datasets/
│   ├── IHDP/csv/      # IHDP数据集
│   ├── IHDP1000/      # 扩展数据集
│   └── TWINS/         # TWINS数据集
├── record/            # 实验记录
│   ├── setup_guide.md # 本文档
│   └── experiment_analysis.md
└── README.md          # 项目说明
```

---

## 预期运行时间

| 硬件 | 单次实验 | 10次重复 |
|------|----------|----------|
| CPU (现代i7) | ~1-2分钟 | ~15-20分钟 |
| GPU (如有支持) | ~30秒 | ~5分钟 |

---

## 验证成功运行

成功运行后，输出应包含：

```
Replication 1/10
Improved validation bound, old: -inf, new: -19.001
Epoch: 1/100, log p(x) >= 23.595, ite_tr: 1.333, ate_tr: 0.329, ...
Improved validation bound, old: -19.001, new: -15.994
Epoch: 11/100, log p(x) >= 17.682, ite_tr: 1.196, ate_tr: 0.053, ...
...

Replication: 1/10, tr_ite: 1.170, tr_ate: 0.010, tr_pehe: 0.688, te_ite: 0.970, te_ate: 0.157, te_pehe: 0.832
...

CEVAE model total scores
train ITE: 1.980+-0.660, train ATE: 0.475+-0.211, train PEHE: 2.473+-1.351
test ITE: 1.531+-0.355, test ATE: 0.340+-0.110, test PEHE: 2.023+-0.969
```

---

## 环境导出与迁移

### 导出环境

```bash
conda env export -n cevae > cevae_environment.yml
```

### 在其他机器恢复

```bash
conda env create -f cevae_environment.yml
```

---

## 参考资源

- **论文**: https://arxiv.org/abs/1705.08821
- **Edward文档**: http://edwardlib.org/
- **TensorFlow 1.1文档**: https://www.tensorflow.org/versions/r1.1/

---

**文档版本**: 1.0
**最后更新**: 2026-03-12
**测试环境**: Windows 11, Miniconda3, Python 3.5.6
