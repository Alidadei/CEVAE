# CEVAE多数据集训练操作指南

**最后更新**: 2026-03-13
**适用数据集**: IHDP, IHDP1000, TWINS

---

## 快速开始

### 激活环境

```bash
# Windows
conda activate cevae

# Linux/Mac
source activate cevae
```

### 基本运行命令

#### 方式一：直接运行（结果仅输出到终端）

```bash
# IHDP数据集 (默认，推荐新手)
python cevae_ihdp.py

# IHDP1000数据集 (大规模)
python cevae_ihdp.py -dataset ihdp1000 -epochs 50

# TWINS数据集 (真实数据)
python cevae_ihdp.py -dataset twins
```

#### 方式二：自动记录结果（推荐）⭐

**重要实验请使用自动记录，结果会保存到 `record/` 目录**

```bash
# 使用自动记录工具
python run_with_log.py -dataset ihdp

# IHDP1000 分离模式（推荐配置）
python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 10

# 带自定义参数
python run_with_log.py -dataset ihdp -reps 20 -epochs 150 -lr 0.0005
```

**Windows 批处理方式**:
```bash
run_with_log.bat ihdp
run_with_log.bat ihdp1000 -separate_reps -n_reps 10
```

**结果文件自动命名**:
```
record/ihdp_standard_20260313_143000.txt
record/ihdp1000_separate_20260313_150000.txt
```

> 💡 **提示**: 详细说明见第十一章《实验结果记录》

---

## 一、IHDP数据集训练

### 1.1 基本信息

| 项目 | 说明 |
|------|------|
| 样本数 | 747 × 10次重复 |
| 训练时间 | ~1分钟 (100 epochs) |
| 难度 | ⭐ 简单 |
| 推荐场景 | 快速验证、学习使用 |

### 1.2 运行命令

**默认运行** (10次重复，100个epoch):
```bash
python cevae_ihdp.py -dataset ihdp
```

**自定义参数**:
```bash
# 减少重复次数 (快速测试)
python cevae_ihdp.py -dataset ihdp -reps 5

# 调整训练轮数
python cevae_ihdp.py -dataset ihdp -epochs 50

# 调整学习率
python cevae_ihdp.py -dataset ihdp -lr 0.0005

# 调整早停检查频率
python cevae_ihdp.py -dataset ihdp -earl 5

# 调整输出频率
python cevae_ihdp.py -dataset ihdp -print_every 5
```

**完整参数示例**:
```bash
python cevae_ihdp.py -dataset ihdp -reps 20 -epochs 150 -lr 0.001 -earl 10 -print_every 10
```

### 1.3 预期输出

```
Replication 1/10
Improved validation bound, old: -inf, new: -19.001
Epoch: 1/100, log p(x) >= 23.595, ite_tr: 1.333, ate_tr: 0.329, pehe_tr: 1.154, ...
Improved validation bound, old: -19.001, new: -15.994
...
Replication: 1/10, tr_ite: 1.170, tr_ate: 0.010, tr_pehe: 0.688, te_ite: 0.970, ...
...

Replication 10/10
...

CEVAE model total scores on IHDP
train ITE: 1.980+-0.660, train ATE: 0.475+-0.211, train PEHE: 2.473+-1.351
test ITE: 1.531+-0.355, test ATE: 0.340+-0.110, test PEHE: 2.023+-0.969
```

### 1.4 模型保存位置

```
models/cevae_ihdp/
├── checkpoint
├── cevae_ihdp.meta
├── cevae_ihdp.data-00000-of-00001
└── cevae_ihdp.index
```

---

## 二、IHDP1000数据集训练

### 2.1 基本信息

| 项目 | 说明 |
|------|------|
| 样本数 | 1000个replications，每个672训练样本+75测试样本 |
| 训练时间 | 取决于模式（见下文） |
| 难度 | ⭐⭐⭐ 中等 |
| 推荐场景 | 大规模性能测试、模型评估 |

### 2.2 两种使用模式

IHDP1000 有两种使用模式，**强烈推荐使用分离模式进行正式评估**：

#### 模式对比

| 特性 | 合并模式 (默认) | 分离模式 (推荐) |
|------|-----------------|----------------|
| **数据使用** | 所有1000个replications合并为一个大训练集 | 每个replication独立训练 |
| **训练样本** | ~600,000 | 每个replication ~470 |
| **模型数量** | 1个 | N个 (每个replication一个) |
| **训练时间** | ~30-60分钟 (单次训练) | 取决于replications数量 |
| **输出格式** | 单次结果 (无标准差) | 平均值±标准差 |
| **ITE准确性** | ✗ 差 (约14) | ✓ 好 (约2) |
| **适用场景** | 快速测试、可扩展性验证 | 正式研究、论文发表 |

#### 合并模式 (Combined Mode)

**特点**: 所有数据合并训练一次，快速但个体预测不准确

```bash
# 默认运行 (合并模式)
python cevae_ihdp.py -dataset ihdp1000

# 减少epoch数
python cevae_ihdp.py -dataset ihdp1000 -epochs 50 -earl 5 -print_every 5

# 快速测试 (验证代码正确性)
python cevae_ihdp.py -dataset ihdp1000 -epochs 10 -earl 2 -print_every 2
```

**预期输出**:
```
Using IHDP1000 in combined mode (all replications merged)
...
CEVAE model total scores on IHDP1000
train ITE: 14.530, train ATE: 0.009, train PEHE: 9.945
test ITE: 14.697, test ATE: 0.018, test PEHE: 10.065
```

⚠️ **注意**: 合并模式的 ITE 和 PEHE 误差很大，这是因为不同 replications 之间因果机制差异导致的异质性。详见 `docs/ihdp1000_modes_comparison.md`。

#### 分离模式 (Separate Replications Mode) - **推荐**

**特点**: 每个replication独立训练，结果准确但耗时较长

```bash
# 推荐: 10个replications (快速验证)
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 10

# 标准评估: 100个replications (平衡速度和稳定性)
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 100

# 完整评估: 1000个replications (最全面，耗时很长)
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 1000
```

**预期输出**:
```
Using IHDP1000 in separate replications mode: 10 replications

Replication 1/10
...
Replication: 1/10, tr_ite: 2.15, tr_ate: 0.31, tr_pehe: 2.78, ...

Replication 10/10
...

CEVAE model total scores on IHDP1000
train ITE: 1.952+-0.635, train ATE: 0.240+-0.063, train PEHE: 2.472+-1.325
test ITE: 1.813+-0.435, test ATE: 0.407+-0.124, test PEHE: 2.082+-0.882
```

✓ **优点**: 结果与 IHDP 基准一致，有标准差可以评估模型稳定性

### 2.3 推荐使用策略

```
┌─────────────────────────────────────────────────────────────────┐
│                      使用场景决策                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  快速验证代码 → 合并模式, 10 epochs                             │
│  测试模型可扩展性 → 合并模式                                     │
│                                                                 │
│  正式研究/论文 → 分离模式, 100 replications                     │
│  与其他方法比较 → 分离模式                                       │
│  需要稳定性评估 → 分离模式 (有标准差)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 训练进度监控

#### 合并模式

IHDP1000数据量大，每个epoch需要30-60秒。正常输出：

```
Replication 1/1
epoch #0| 100%|##################################################|ETA:  0:00:45
epoch #1| 100%|##################################################|ETA:  0:00:43
...
Improved validation bound, old: -inf, new: -XXX.XXX
Epoch: 5/50, log p(x) >= XX.XXX, ...
```

#### 分离模式

每个replication独立训练，可以看到进度：

```
Using IHDP1000 in separate replications mode: 10 replications

Replication 1/10
epoch #0| 100%|##################################################|ETA:  0:00:05
...
Replication: 1/10, tr_ite: 2.15, ...

Replication 2/10
...

Replication 10/10
...
```

每个replication训练约10-15秒（取决于epochs设置）。

### 2.5 新增参数说明

| 参数 | 说明 | 仅适用于 |
|------|------|----------|
| `-separate_reps` | 启用分离模式（每个replication独立训练） | IHDP1000 |
| `-n_reps` | 指定使用的replications数量（默认100） | IHDP1000 |

### 2.6 如果中断训练

按 `Ctrl+C` 可以安全中断：
- **合并模式**: 已训练的epoch不会保存
- **分离模式**: 已完成的replications结果会保留

### 2.7 模型保存位置

```
models/cevae_ihdp1000/
├── checkpoint
├── cevae_ihdp1000.meta
├── cevae_ihdp1000.data-00000-of-00001
└── cevae_ihdp1000.index
```

分离模式下，每个replication会覆盖保存同一个模型文件。

---

## 三、TWINS数据集训练

### 3.1 基本信息

| 项目 | 说明 |
|------|------|
| 样本数 | ~13,000对双胞胎 |
| 训练时间 | ~5-10分钟 (100 epochs) |
| 难度 | ⭐⭐ 简单 |
| 特殊说明 | 无反事实数据，无法计算ITE/ATE/PEHE |

### 3.2 运行命令

**标准运行**:
```bash
python cevae_ihdp.py -dataset twins
```

**调整参数**:
```bash
python cevae_ihdp.py -dataset twins -epochs 150 -lr 0.001
```

### 3.3 预期输出

```
Warning: Dataset has no counterfactuals, only computing basic metrics
Replication 1/1
...
CEVAE model total scores on TWINS
train ITE: X.XXX+-X.XXX, train ATE: X.XXX+-X.XXX, train PEHE: X.XXX+-X.XXX
test ITE: X.XXX+-X.XXX, test ATE: X.XXX+-X.XXX, test PEHE: X.XXX+-X.XXX
```

**注意**: TWINS无反事实数据，返回的ITE/ATE/PEHE值都是RMSE占位符，不代表真实的因果效应误差。

### 3.4 模型保存位置

```
models/cevae_twins/
├── checkpoint
├── cevae_twins.meta
├── cevae_twins.data-00000-of-00001
└── cevae_twins.index
```

---

## 四、命令行参数完整说明

### 4.1 参数列表

| 参数 | 默认值 | 说明 | 适用数据集 |
|------|--------|------|-----------|
| `-dataset` | `ihdp` | 选择数据集 (ihdp/ihdp1000/twins) | 所有 |
| `-reps` | `10` | 重复实验次数 | 仅IHDP |
| `-epochs` | `100` | 训练轮数 | 所有 |
| `-lr` | `0.001` | 学习率 | 所有 |
| `-earl` | `10` | 早停检查频率（每隔N个epoch检查验证集并保存最佳模型） | 所有 |
| `-print_every` | `10` | 输出训练统计的频率（每隔N个epoch显示ITE/ATE/PEHE等指标） | 所有 |
| `-opt` | `adam` | 优化器 (adam/adamax) | 所有 |
| `-separate_reps` | `False` | 启用IHDP1000分离模式 | 仅IHDP1000 |
| `-n_reps` | `None` | IHDP1000使用的replications数量 | 仅IHDP1000 |

### 4.2.1 `-earl` 和 `-print_every` 详细说明

#### `-earl` 早停检查频率

**作用**: 每隔多少个 epoch 检查一次验证集性能，如果模型变好就保存

**工作原理**:
```
Epoch 1-4:  训练...
Epoch 5:    训练完成
            ↓ 检查验证集性能
            ↓ 如果更好 → 保存模型
            输出: "Improved validation bound, old: X, new: Y"

Epoch 6-9:  训练...
Epoch 10:   训练完成
            ↓ 再次检查验证集性能
            ...
```

**设置建议**:
| 设置 | 适用场景 | 说明 |
|------|----------|------|
| `-earl 2` | 快速测试 | 更频繁检查和保存 |
| `-earl 5` | 长时间训练 | 平衡性能和安全 |
| `-earl 10` | 默认/正常训练 | 标准配置 |

#### `-print_every` 输出频率

**作用**: 每隔多少个 epoch 输出一次详细的训练统计信息

**输出内容**:
```
Epoch: 10/100, log p(x) >= 23.595, ite_tr: 1.333, ate_tr: 0.329,
       pehe_tr: 1.154, rmse_f_tr: 0.876, rmse_cf_tr: 1.234,
       ite_te: 1.123, ate_te: 0.287, pehe_te: 1.045, dt: 3.456
```

**指标说明**:
- `ite_tr/te`: 个体处理效应误差（训练/测试）
- `ate_tr/te`: 平均处理效应误差（训练/测试）
- `pehe_tr/te`: 异质效应误差（训练/测试）
- `rmse_f_tr`: 事实结果RMSE（训练）
- `rmse_cf_tr`: 反事实结果RMSE（训练）
- `dt`: 单个epoch耗时（秒）

**设置建议**:
| 设置 | 适用场景 | 说明 |
|------|----------|------|
| `-print_every 1` | 调试模式 | 每个epoch都输出，日志很多 |
| `-print_every 5` | 快速测试 | 适中的输出频率 |
| `-print_every 10` | 默认/正常训练 | 标准配置 |
| `-print_every 20` | 减少日志 | 输出较少，训练略快 |

#### 两个参数的区别

| 特性 | `-earl` | `-print_every` |
|------|---------|----------------|
| **作用** | 检查并保存模型 | 显示训练进度 |
| **输出内容** | 验证边界改善信息 | ITE/ATE/PEHE等详细统计 |
| **影响** | 模型保存频率 | 日志详细程度 |

#### 实际例子对比

**例子1: 默认设置（推荐）**
```bash
python cevae_ihdp.py -dataset ihdp -earl 10 -print_every 10
```
输出:
```
Epoch: 10/100, ite_tr: 1.5, ate_tr: 0.3, pehe_tr: 1.1, ...
Improved validation bound, old: -inf, new: -19.001

Epoch: 20/100, ite_tr: 1.2, ate_tr: 0.25, pehe_tr: 0.9, ...
Improved validation bound, old: -19.001, new: -18.500
```

**例子2: 更频繁的输出（适合观察训练过程）**
```bash
python cevae_ihdp.py -dataset ihdp -earl 5 -print_every 5
```
输出:
```
Epoch: 5/100, ite_tr: 1.8, ate_tr: 0.35, pehe_tr: 1.3, ...
Improved validation bound, old: -inf, new: -19.001

Epoch: 10/100, ite_tr: 1.6, ate_tr: 0.32, pehe_tr: 1.2, ...

Epoch: 15/100, ite_tr: 1.4, ate_tr: 0.28, pehe_tr: 1.0, ...
Improved validation bound, old: -19.001, new: -18.800
```

### 4.2 参数使用建议

#### 新手快速验证
```bash
python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20 -print_every 5
```

#### 标准实验 (论文复现)
```bash
# IHDP标准实验
python cevae_ihdp.py -dataset ihdp -reps 10 -epochs 100

# IHDP1000推荐配置 (分离模式)
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 100
```

#### 大规模数据集快速测试
```bash
# IHDP1000合并模式 (快速)
python cevae_ihdp.py -dataset ihdp1000 -epochs 50 -earl 5 -print_every 5

# IHDP1000分离模式 (快速验证)
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 10
```

#### 调试模式
```bash
python cevae_ihdp.py -dataset ihdp -epochs 5 -print_every 1
```

---

## 五、运行流程图

```
开始：!先正确进入项目根目录 X:\XXX\CEVAE
  │
  ├─> 激活环境: conda activate cevae
  │
  ├─> 选择数据集
  │     ├─> IHDP (小数据集，快速)
  │     ├─> IHDP1000 (大数据集，耗时)
  │     └─> TWINS (真实数据，无反事实)
  │
  ├─> 运行训练
  │     └─> python cevae_ihdp.py -dataset [数据集名]
  │
  ├─> 观察输出
  │     ├─> 训练进度条
  │     ├─> 验证边界改善
  │     └─> 每10个epoch的统计信息
  │
  ├─> 训练完成
  │     ├─> 自动加载最佳模型
  │     ├─> 计算最终评估指标
  │     └─> 输出汇总结果
  │
  └─> 查看模型
        └─> models/cevae_[数据集名]/
```

---

## 六、常见问题

### 6.1 如何知道训练是否正常？

**正常的训练输出应该包含**:
1. 进度条显示百分比和ETA
2. 每10个epoch输出一次统计信息
3. 看到 "Improved validation bound" 说明模型在改善

**异常信号**:
- loss变成NaN → 学习率可能太大，尝试 `-lr 0.0005`
- 长时间无输出 → 可能卡住，检查数据加载

### 6.2 训练中断怎么办？

代码会自动保存验证集上最佳的模型。即使中断：
- 已保存的checkpoint不受影响
- 重新运行会从头开始（不会断点续训）

### 6.3 如何比较不同数据集的结果？

结果输出包含数据集名称：
```
CEVAE model total scores on IHDP
...
CEVAE model total scores on IHDP1000
...
CEVAE model total scores on TWINS
...
```

### 6.4 内存不足怎么办？

IHDP1000数据量大，如果遇到内存错误：
```bash
# 减小batch size (需要修改代码)
# 或者减少训练数据量
```

### 6.5 如何并行运行多个实验？

在多个终端/命令行窗口中：
```bash
# 终端1
python cevae_ihdp.py -dataset ihdp

# 终端2
python cevae_ihdp.py -dataset twins

# 终端3
python cevae_ihdp.py -dataset ihdp1000 -epochs 50
```

每个实验会保存到独立的模型目录。

---

## 七、输出指标说明

### 7.1 训练过程中的指标

| 指标 | 含义 | 越小越好 |
|------|------|----------|
| `log p(x)` | 负对数似然损失 | ✓ |
| `ite_tr/te` | 个体处理效应误差(训练/测试) | ✓ |
| `ate_tr/te` | 平均处理效应误差(训练/测试) | ✓ |
| `pehe_tr/te` | 异质效应误差(训练/测试) | ✓ |
| `rmse_f_tr` | 事实结果RMSE(训练) | ✓ |
| `rmse_cf_tr` | 反事实结果RMSE(训练) | ✓ |
| `dt` | 单个epoch耗时(秒) | ✓ |

### 7.2 最终汇总指标

```
train ITE: 1.980+-0.660
        ↑均值   ↑标准误

test PEHE: 2.023+-0.969
       ↑均值   ↑标准误
```

**标准误小的说明**: 多次重复实验结果一致，模型稳定。

---

## 八、推荐的工作流程

### 初次使用 (学习流程)

```bash
# 1. 快速验证 (~1分钟)
python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20

# 2. 标准实验 (~1分钟)
python cevae_ihdp.py -dataset ihdp -reps 10 -epochs 100

# 3. 查看结果
ls models/cevae_ihdp/
```

### 完整实验 (对比三个数据集)

```bash
# 步骤1: IHDP (~1分钟)
python cevae_ihdp.py -dataset ihdp -reps 10

# 步骤2: TWINS (~5分钟)
python cevae_ihdp.py -dataset twins

# 步骤3: IHDP1000 (~30-60分钟)
python cevae_ihdp.py -dataset ihdp1000 -epochs 50

# 步骤4: 查看所有模型
ls models/
```

### 高级用户 (参数调优)

```bash
# 尝试不同学习率
python cevae_ihdp.py -dataset ihdp -lr 0.0005 -reps 5
python cevae_ihdp.py -dataset ihdp -lr 0.002 -reps 5

# 尝试不同优化器
python cevae_ihdp.py -dataset ihdp -opt adamax

# 尝试更多训练轮数
python cevae_ihdp.py -dataset ihdp -epochs 200
```

---

## 九、文件管理

### 9.1 模型文件结构

```
models/
├── cevae_ihdp/              # IHDP模型
│   ├── checkpoint
│   ├── cevae_ihdp.meta       # 958KB - 模型结构
│   ├── cevae_ihdp.data-*      # 6.6MB - 模型权重
│   └── cevae_ihdp.index      # 6.8KB - 索引
│
├── cevae_ihdp1000/          # IHDP1000模型
│   └── ...
│
└── cevae_twins/             # TWINS模型
    └── ...
```

### 9.2 清理旧模型

```bash
# 删除特定数据集的模型
rm -rf models/cevae_ihdp1000/

# 删除所有模型
rm -rf models/cevae_*

# 备份模型
cp -r models/cevae_ihdp models_backup/
```

---

## 十、快速参考卡片

```bash
# ============ 自动记录实验结果 (推荐) ============
python run_with_log.py -dataset ihdp
python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 10

# Windows 批处理方式
run_with_log.bat ihdp
run_with_log.bat ihdp1000 -separate_reps -n_reps 10

# ============ 直接运行 (不记录) ============
# IHDP
python cevae_ihdp.py -dataset ihdp
python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20

# IHDP1000
python cevae_ihdp.py -dataset ihdp1000 -epochs 50
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 10
python cevae_ihdp.py -dataset ihdp1000 -separate_reps -n_reps 100

# TWINS
python cevae_ihdp.py -dataset twins

# ============ 通用参数 ==========
-reps           # 重复次数 (仅IHDP)
-epochs         # 训练轮数
-lr             # 学习率 (默认0.001)
-earl           # 早停检查频率
-print_every    # 输出频率
-separate_reps  # IHDP1000分离模式
-n_reps         # IHDP1000使用replications数量

# ============ 查看模型 ============
ls models/

# ============ 查看实验记录 ============
ls record/
```

### 模式选择速查

| 场景 | 推荐命令 |
|------|----------|
| IHDP快速验证 | `python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20` |
| IHDP标准实验 | `python run_with_log.py -dataset ihdp -reps 10 -epochs 100` |
| IHDP1000快速测试 | `python cevae_ihdp.py -dataset ihdp1000 -epochs 10` |
| IHDP1000正式评估 | `python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 100` |

### 命令选择建议

```
┌─────────────────────────────────────────────────────────────┐
│                  运行方式选择                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  快速测试/调试        → 直接运行 (python cevae_ihdp.py)     │
│  重要实验/论文研究    → 自动记录 (python run_with_log.py)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 十一、实验结果记录

### 11.1 为什么需要记录实验结果？

实验记录可以帮助你：
- 追踪不同配置的实验结果
- 对比不同参数的效果
- 复现已有的实验结果
- 为论文或报告提供数据支持

### 11.2 自动记录（推荐）

项目提供了自动记录工具，可以自动保存实验结果到 `record/` 目录。

#### 方式一：使用 Python 脚本

```bash
# 基本用法
python run_with_log.py -dataset ihdp

# IHDP1000 分离模式
python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 10

# 带自定义参数
python run_with_log.py -dataset ihdp -reps 20 -epochs 150 -lr 0.0005
```

**自动生成文件名示例**:
```
record/ihdp_standard_20260313_143000.txt
record/ihdp1000_separate_20260313_150000.txt
record/ihdp1000_combined_20260313_160000.txt
```

#### 方式二：使用批处理脚本（Windows）

```bash
# 基本用法
run_with_log.bat ihdp

# IHDP1000 分离模式
run_with_log.bat ihdp1000 -separate_reps -n_reps 10
```

#### 方式三：使用 tee 命令

```bash
# Linux/Mac 或 Windows Git Bash
python cevae_ihdp.py -dataset ihdp | tee record/ihdp_standard_$(date +%Y%m%d_%H%M%S).txt

# Windows PowerShell
python cevae_ihdp.py -dataset ihdp | Tee-Object -FilePath "record/ihdp_standard_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
```

### 11.3 手动记录

如果自动记录工具不可用，可以手动记录：

#### 步骤

1. **运行训练程序**
   ```bash
   python cevae_ihdp.py -dataset ihdp
   ```

2. **复制完整的终端输出**

3. **创建记录文件**
   ```bash
   # 复制模板
   cp record/template.txt record/ihdp_standard_20260313_143000.txt

   # 或直接创建新文件
   notepad record/ihdp_standard_20260313_143000.txt
   ```

4. **填写实验信息**
   - 使用提供的模板 `record/template.txt`
   - 填写实验配置、粘贴输出、记录分析

### 11.4 记录文件格式

每个记录文件应包含：

```yaml
================================================================================
CEVAE 实验记录
================================================================================

【实验配置】
数据集: IHDP
模式: standard
Replications: 10
Epochs: 100
学习率: 0.001
时间: 2026-03-13 14:30:00

【训练过程】
[粘贴关键输出]

【最终结果】
CEVAE model total scores on IHDP
train ITE: 1.980+-0.660
train ATE: 0.475+-0.211
train PEHE: 2.473+-1.351
test ITE: 1.531+-0.355
test ATE: 0.340+-0.110
test PEHE: 2.023+-0.969

【分析】
[记录你的观察和结论]
================================================================================
```

### 11.5 record 目录结构

```
record/
├── README.md                    # 使用说明
├── template.txt                 # 实验记录模板
│
├── ihdp_standard_20260313_143000.txt     # IHDP 标准实验
├── ihdp1000_combined_20260313_150000.txt  # IHDP1000 合并模式
├── ihdp1000_separate_20260313_160000.txt  # IHDP1000 分离模式
│
└── experiments_summary.xlsx     # 可选: 实验汇总表格
```

### 11.6 实验对比技巧

#### 创建对比表格

在 `record/` 目录创建 `experiments_summary.xlsx` 或 `summary.csv`：

| 实验日期 | 数据集 | 模式 | ITE | ATE | PEHE | 备注 |
|---------|--------|------|-----|-----|------|------|
| 03-13 | ihdp | standard | 1.53 | 0.34 | 2.02 | 基准实验 |
| 03-13 | ihdp1000 | combined | 14.70 | 0.02 | 10.07 | ❌ 不推荐 |
| 03-13 | ihdp1000 | separate | 1.81 | 0.41 | 2.08 | ✓ 推荐 |

#### 对比命令

```bash
# 查看所有 IHDP 实验结果
cat record/ihdp_*.txt | grep "test ITE"

# 查看最新的实验结果
ls -lt record/*.txt | head -1 | xargs cat
```

### 11.7 推荐的记录习惯

```
┌─────────────────────────────────────────────────────────────┐
│                  实验记录最佳实践                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ 每次重要实验都记录                                       │
│  ✓ 使用自动记录工具                                         │
│  ✓ 添加简短的分析和结论                                      │
│  ✓ 定期整理和对比结果                                        │
│  ✓ 备份 record 目录                                          │
│                                                             │
│  ✗ 不要只保存最终结果（缺少过程信息）                         │
│  ✗ 不要使用混乱的文件名                                      │
│  ✗ 不要在文件名中使用空格或特殊字符                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**文档版本**: 1.4
**最后更新**: 2026-03-13
**更新内容**:
- 添加 IHDP1000 分离模式说明
- 添加实验结果自动记录功能
- 快速开始部分更新，突出自动记录工具
- 新增 `-earl` 和 `-print_every` 参数详细说明
