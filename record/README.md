# 实验结果记录目录

本目录用于保存主要训练程序的运行结果。

## 文件命名规范

```
{数据集}_{模式}_{时间}.txt
```

例如：
- `ihdp_standard_20260313_143000.txt`
- `ihdp1000_separate_10reps_20260313_150000.txt`
- `ihdp1000_combined_20260313_160000.txt`

## 记录内容模板

每次实验记录应包含：

### 1. 实验配置
```yaml
数据集: IHDP/IHDP1000/TWINS
模式: standard/combined/separate
Replications: 10/100/1000
Epochs: 100
学习率: 0.001
优化器: adam/adamax
时间: YYYY-MM-DD HH:MM:SS
```

### 2. 训练过程
- 每个 epoch 的日志输出
- 验证边界改善情况
- 训练时间

### 3. 最终结果
```
CEVAE model total scores on {DATASET}
train ITE: X.XXX±X.XXX
train ATE: X.XXX±X.XXX
train PEHE: X.XXX±X.XXX
test ITE: X.XXX±X.XXX
test ATE: X.XXX±X.XXX
test PEHE: X.XXX±X.XXX
```

## 自动记录工具

使用以下命令自动记录实验结果：

```bash
# 方式1: 使用 tee 命令（推荐）
python cevae_ihdp.py -dataset ihdp | tee record/ihdp_standard_$(date +%Y%m%d_%H%M%S).txt

# 方式2: 先运行后手动保存
python cevae_ihdp.py -dataset ihdp > record/ihdp_standard_$(date +%Y%m%d_%H%M%S).txt
```

## 手动记录步骤

1. 运行训练程序
2. 复制完整的终端输出
3. 在 `record/` 目录创建新文件，使用标准命名
4. 粘贴输出内容
5. 添加必要的实验配置信息

## 文件列表

| 文件名 | 数据集 | 模式 | 日期 | 说明 |
|--------|--------|------|------|------|
