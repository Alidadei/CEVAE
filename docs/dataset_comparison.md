# CEVAE多数据集对比分析报告

**日期**: 2026-03-12
**项目**: Causal Effect Variational Autoencoder (CEVAE)
**目的**: 在IHDP、IHDP1000、TWINS三个数据集上测试CEVAE模型性能

---

## 一、数据集对比分析

### 1.1 基本信息对比

| 特征 | IHDP | IHDP1000 | TWINS |
|------|------|----------|-------|
| **全称** | Infant Health and Development Program | IHDP扩展版 | Twins Birth Outcomes |
| **样本数** | 747 × 10 replications | ~7,459 × 1000 = 7,459,000 | ~13,000对双胞胎 |
| **协变量维度** | 25 (6连续 + 19二值) | 25 (6连续 + 19二值) | 50 (混合类型) |
| **数据类型** | 合成数据 | 合成数据 | 真实数据 |
| **反事实数据** | ✓ 有 | ✓ 有 | ✗ 无 |
| **文件格式** | CSV | .npy (numpy) | CSV |
| **数据分割** | 需手动分割 | 预分割 train/test | 需手动分割 |

### 1.2 数据结构详解

#### IHDP
```
 ihdp_npci_1.csv (747 rows):
 列0: t (treatment)
 列1: y (factual outcome)
 列2: y_cf (counterfactual outcome)
 列3: mu_0 (control potential outcome)
 列4: mu_1 (treated potential outcome)
 列5-29: x (covariates)
```

#### IHDP1000
```
 ihdp_npci_1-1000.train/:
 - x.npy: (672, 25, 1000) - 672个基础样本 × 1000次重复
 - t.npy: (672, 1000) - 处理变量
 - yf.npy: (672, 1000) - 事实结果
 - ycf.npy: (672, 1000) - 反事实结果
 - mu0.npy: (672, 1000) - 控制组潜在结果
 - mu1.npy: (672, 1000) - 处理组潜在结果

总训练样本: 672,000
总测试样本: 750,000
```

#### TWINS
```
 twin_pairs_X_3years_samesex.csv: 协变量
 twin_pairs_T_3years_samesex.csv: 处理变量
 twin_pairs_Y_3years_samesex.csv: 结果变量(mort_0, mort_1)

注意: 每行代表一对双胞胎，两个结果分别对应两个双胞胎
```

### 1.3 数据规模对比

```
IHDP:      ~7,000    样本
IHDP1000:  ~1,420,000 样本 (200倍于IHDP)
TWINS:     ~13,000   样本 (双胞胎对)
```

---

## 二、代码修改说明

### 2.1 新增数据集加载类

#### datasets.py 新增内容

**IHDP1000类** (第37-113行)
- 处理3维numpy数组 (n_samples, n_features, n_replications)
- 数据重塑和转置以匹配模型输入格式
- 特征类型继承自IHDP

**TWINS类** (第116-177行)
- 从3个独立CSV文件加载数据
- 自动检测二值/连续特征
- 处理缺失值
- 无反事实数据时的特殊处理

### 2.2 主程序修改 (cevae_ihdp.py)

```python
# 新增命令行参数
-parser.add_argument('-dataset', choices=['ihdp', 'ihdp1000', 'twins'], default='ihdp')

# 动态数据集选择
if args.dataset == 'ihdp':
    dataset = IHDP(replications=args.reps)
elif args.dataset == 'ihdp1000':
    dataset = IHDP1000()
elif args.dataset == 'twins':
    dataset = TWINS()

# 无反事实数据时的评估器处理
if has_counterfactuals:
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)
else:
    evaluator_test = Evaluator(yte, tte)
```

### 2.3 评估器修改 (evaluation.py)

```python
def calc_stats(self, ypred1, ypred0):
    if self.mu0 is None or self.mu1 is None:
        # 对于无反事实的数据集，返回RMSE作为占位符
        ypred = (1 - self.t) * ypred0 + self.t * ypred1
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        return rmse_factual, rmse_factual, rmse_factual
    # ... 正常计算
```

---

## 三、运行命令

### 3.1 IHDP (原始数据集)

```bash
# 默认运行 (10次重复)
python cevae_ihdp.py -dataset ihdp

# 自定义重复次数
python cevae_ihdp.py -dataset ihdp -reps 50
```

### 3.2 IHDP1000 (大规模数据集)

```bash
# 标准运行 (推荐: 减少epoch数)
python cevae_ihdp.py -dataset ihdp1000 -epochs 50 -earl 5 -print_every 5

# 快速测试
python cevae_ihdp.py -dataset ihdp1000 -epochs 10 -earl 2
```

### 3.3 TWINS (真实数据集)

```bash
# 标准运行
python cevae_ihdp.py -dataset twins -epochs 100

# 注意: TWINS无反事实数据，无法计算ITE/ATE/PEHE
# 只能计算预测RMSE
```

---

## 四、预期运行时间

| 数据集 | 单epoch时间 | 总时间(100 epochs) | 内存占用 |
|--------|------------|-------------------|----------|
| IHDP | ~0.5秒 | ~1分钟 | <1GB |
| IHDP1000 | ~30-60秒 | ~1-2小时 | ~4GB |
| TWINS | ~2-5秒 | ~5-10分钟 | ~1GB |

---

## 五、实验结果汇总

### 5.1 IHDP (10次重复)

```
CEVAE model total scores on IHDP
train ITE: 1.980+-0.660, train ATE: 0.475+-0.211, train PEHE: 2.473+-1.351
test ITE: 1.531+-0.355, test ATE: 0.340+-0.110, test PEHE: 2.023+-0.969
```

**分析**:
- 9/10次实验表现良好
- 实验9出现异常（PEHE > 10）
- 去除异常后: 测试PEHE ≈ 0.86

### 5.2 IHDP1000

**状态**: 正在训练中...

**注意**:
- 数据量大，训练时间长
- 可能需要调整batch size或学习率
- 建议使用更多计算资源

### 5.3 TWINS

**状态**: 待测试

**限制**:
- 无反事实数据，无法计算标准因果推断指标
- 只能评估预测性能（RMSE）
- 结果不能直接与IHDP/IHDP1000比较

---

## 六、技术挑战与解决方案

### 6.1 IHDP1000数据重塑问题

**问题**:
```
ValueError: Cannot feed value of shape (100, 19, 1000) for Tensor 'x_bin:0', which has shape '(?, 19)'
```

**原因**: 数据结构为 (n_samples, n_features, n_replications)

**解决**:
```python
# 转置并重塑
x_train = x_train.transpose(0, 2, 1).reshape(-1, n_features)
```

### 6.2 TWINS无反事实数据

**问题**: 评估指标需要mu0/mu1

**解决**: 返回RMSE作为占位符，或使用其他评估方法

### 6.3 Python编码问题

**问题**: SyntaxError due to smart quotes

**解决**: 使用标准ASCII引号 (")

---

## 七、下一步工作

### 7.1 待完成

- [ ] 完成IHDP1000训练
- [ ] 测试TWINS数据集
- [ ] 编写完整对比报告
- [ ] 优化IHDP1000训练速度

### 7.2 优化建议

1. **IHDP1000优化**
   - 增加batch size (当前100 → 500-1000)
   - 使用学习率衰减
   - 考虑分布式训练

2. **TWINS评估**
   - 实现无反事实数据的评估指标
   - 与基线方法比较

3. **代码改进**
   - 添加数据集自动验证
   - 统一接口设计
   - 添加进度保存/恢复功能

---

## 八、文件结构

```
CEVAE/
├── cevae_ihdp.py          # 主程序 (已修改，支持多数据集)
├── datasets.py            # 数据集加载器 (已添加IHDP1000, TWINS类)
├── evaluation.py          # 评估器 (已修改，支持无反事实数据)
├── datasets/
│   ├── IHDP/csv/          # IHDP数据 (10个CSV文件)
│   ├── IHDP1000/          # IHDP1000数据 (.npy格式)
│   │   ├── ihdp_npci_1-1000.train/
│   │   └── ihdp_npci_1-1000.test/
│   └── TWINS/             # TWINS数据
│       ├── twin_pairs_X_*.csv
│       ├── twin_pairs_T_*.csv
│       └── twin_pairs_Y_*.csv
├── models/                # 模型保存目录
└── record/
    ├── setup_guide.md     # 环境配置攻略
    ├── experiment_analysis.md  # IHDP实验分析
    └── dataset_comparison.md   # 本报告
```

---

**报告生成时间**: 2026-03-12
**版本**: 1.0
**状态**: 进行中
