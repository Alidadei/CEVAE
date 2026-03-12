# CEVAE多数据集训练操作指南

**最后更新**: 2026-03-12
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

```bash
# IHDP数据集 (默认，推荐新手)
python cevae_ihdp.py

# IHDP1000数据集 (大规模)
python cevae_ihdp.py -dataset ihdp1000 -epochs 50

# TWINS数据集 (真实数据)
python cevae_ihdp.py -dataset twins
```

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
| 样本数 | ~1,420,000 (672K训练 + 750K测试) |
| 训练时间 | ~30-60分钟 (100 epochs, CPU) |
| 难度 | ⭐⭐⭐ 中等 |
| 推荐场景 | 大规模性能测试 |

### 2.2 运行命令

**推荐配置** (减少epoch数):
```bash
python cevae_ihdp.py -dataset ihdp1000 -epochs 50 -earl 5 -print_every 5
```

**完整训练** (需要1-2小时):
```bash
python cevae_ihdp.py -dataset ihdp1000 -epochs 100
```

**快速测试** (验证代码正确性):
```bash
python cevae_ihdp.py -dataset ihdp1000 -epochs 10 -earl 2 -print_every 2
```

### 2.3 训练进度监控

IHDP1000数据量大，每个epoch需要30-60秒。正常输出：

```
Replication 1/1
epoch #0| 100%|##################################################|ETA:  0:00:45
epoch #1| 100%|##################################################|ETA:  0:00:43
...
Improved validation bound, old: -inf, new: -XXX.XXX
Epoch: 5/50, log p(x) >= XX.XXX, ...
```

### 2.4 如果中断训练

按 `Ctrl+C` 可以安全中断，但已训练的epoch不会保存。

### 2.5 模型保存位置

```
models/cevae_ihdp1000/
├── checkpoint
├── cevae_ihdp1000.meta
├── cevae_ihdp1000.data-00000-of-00001
└── cevae_ihdp1000.index
```

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
| `-earl` | `10` | 早停检查频率(epoch数) | 所有 |
| `-print_every` | `10` | 输出训练统计的频率 | 所有 |
| `-opt` | `adam` | 优化器 (adam/adamax) | 所有 |

### 4.2 参数使用建议

#### 新手快速验证
```bash
python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20 -print_every 5
```

#### 标准实验 (论文复现)
```bash
python cevae_ihdp.py -dataset ihdp -reps 10 -epochs 100
```

#### 大规模数据集
```bash
python cevae_ihdp.py -dataset ihdp1000 -epochs 50 -earl 5 -print_every 5
```

#### 调试模式
```bash
python cevae_ihdp.py -dataset ihdp -epochs 5 -print_every 1
```

---

## 五、运行流程图

```
开始
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
# ============ IHDP ============
python cevae_ihdp.py -dataset ihdp

# 快速测试 (3次重复, 20轮)
python cevae_ihdp.py -dataset ihdp -reps 3 -epochs 20

# ============ IHDP1000 =========
python cevae_ihdp.py -dataset ihdp1000 -epochs 50

# 快速测试 (10轮)
python cevae_ihdp.py -dataset ihdp1000 -epochs 10

# ============ TWINS ============
python cevae_ihdp.py -dataset twins

# ============ 通用参数 ==========
-reps      # 重复次数 (仅IHDP)
-epochs    # 训练轮数
-lr        # 学习率 (默认0.001)
-earl      # 早停检查频率
-print_every # 输出频率

# ============ 查看模型 ============
ls models/
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-12
