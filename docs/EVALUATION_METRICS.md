# CEVAE 评估指标说明

## 概述

本文档说明 CEVAE 模型训练和测试过程中输出的各项评估指标及其计算方法。

---

## 训练过程输出

### 输出示例

```
Epoch: 10/100, log p(x) >= 1.234, ite_tr: 0.567, ate_tr: 0.123, pehe_tr: 0.456,
rmse_f_tr: 0.234, rmse_cf_tr: 0.345, ite_te: 0.678, ate_te: 0.145, pehe_te: 0.489, dt: 1.234
```

### 参数说明

| 参数 | 含义 | 代码位置 |
|------|------|----------|
| `Epoch` | 当前训练轮次 | cevae_ihdp.py:154 |
| `log p(x)` | ELBO 下界（损失函数） | cevae_ihdp.py:171 |
| `ite_tr` | 训练集 ITE RMSE | evaluation.py:14 |
| `ate_tr` | 训练集 ATE 绝对误差 | evaluation.py:22 |
| `pehe_tr` | 训练集 PEHE | evaluation.py:25 |
| `rmse_f_tr` | 训练集事实结果 RMSE | evaluation.py:34 |
| `rmse_cf_tr` | 训练集反事实结果 RMSE | evaluation.py:35 |
| `ite_te` | 测试集 ITE RMSE | evaluation.py:14 |
| `ate_te` | 测试集 ATE 绝对误差 | evaluation.py:22 |
| `pehe_te` | 测试集 PEHE | evaluation.py:25 |
| `dt` | 每个 epoch 耗时（秒） | cevae_ihdp.py:196 |

---

## 最终输出

### 输出示例

```
CEVAE model total scores
train ITE: 0.567+-0.012, train ATE: 0.123+-0.008, train PEHE: 0.456+-0.015
test ITE: 0.678+-0.018, test ATE: 0.145+-0.010, test PEHE: 0.489+-0.020
```

### 参数说明

| 参数 | 含义 | 代码位置 |
|------|------|----------|
| `train/test` | 训练集/测试集 | - |
| `ITE` | 个体处理效应 RMSE | evaluation.py:39 |
| `ATE` | 平均处理效应绝对误差 | evaluation.py:40 |
| `PEHE` | 异质性效应估计精度 | evaluation.py:41 |
| `±` | 10 次重复实验的标准误差 | cevae_ihdp.py:216 |

---

## 计算公式

### 符号说明

| 符号 | 含义 |
|------|------|
| `x` | 协变量（特征） |
| `t` | 处理变量（0=对照组，1=处理组） |
| `y` | 观测结果 |
| `y_cf` | 反事实结果（真实值，仅合成数据可用） |
| `y0` | 模型预测的未处理结果 p(y\|x,t=0,z) |
| `y1` | 模型预测的已处理结果 p(y\|x,t=1,z) |
| `mu0` | 真实的未处理条件均值 |
| `mu1` | 真实的已处理条件均值 |

---

### 1. ITE (Individual Treatment Effect) RMSE

**含义**：个体层面因果效应估计的均方根误差

**计算公式**：
```python
# 真实个体处理效应
true_ite = mu1 - mu0

# 预测个体处理效应（半参估计）
pred_ite[t==1] = y_observed[t==1] - y_pred0[t==1]  # 处理组：用观测值减去预测的对照组结果
pred_ite[t==0] = y_pred1[t==0] - y_observed[t==0]  # 对照组：用预测的处理组结果减去观测值

# RMSE
ITE_RMSE = sqrt(mean((true_ite - pred_ite)²))
```

**代码位置**：evaluation.py:14-20

---

### 2. ATE (Average Treatment Effect) 绝对误差

**含义**：平均因果效应估计的绝对误差

**计算公式**：
```python
# 真实平均处理效应
true_ate = mean(mu1 - mu0)

# 预测平均处理效应
pred_ate = mean(y_pred1 - y_pred0)

# 绝对误差
ATE_Error = |pred_ate - true_ate|
```

**代码位置**：evaluation.py:22-23

---

### 3. PEHE (Precision in Estimation of Heterogeneous Effect)

**含义**：异质性效应估计精度（处理效应差异的估计精度）

**计算公式**：
```python
PEHE = sqrt(mean(((mu1 - mu0) - (y_pred1 - y_pred0))²))
     = sqrt(mean((true_ite - pred_ite_full)²))
```

**与 ITE 的区别**：
- ITE 使用半参估计（处理组用观测值，对照组用观测值）
- PEHE 使用完整预测（所有样本都用 y_pred1 - y_pred0）

**代码位置**：evaluation.py:25-26

---

### 4. RMSE_factual (事实结果 RMSE)

**含义**：模型对实际观测结果的预测误差

**计算公式**：
```python
# 事实预测：根据实际处理情况选择预测
y_pred = (1 - t) * y0 + t * y1

# RMSE
rmse_factual = sqrt(mean((y_pred - y_observed)²))
```

**代码位置**：evaluation.py:34

---

### 5. RMSE_cfactual (反事实结果 RMSE)

**含义**：模型对反事实结果的预测误差（仅合成数据可计算）

**计算公式**：
```python
# 反事实预测：根据实际处理情况选择相反的预测
y_pred_cf = t * y0 + (1 - t) * y1

# RMSE
rmse_cfactual = sqrt(mean((y_pred_cf - y_cfactual)²))
```

**代码位置**：evaluation.py:35

---

### 6. ELBO (Evidence Lower Bound)

**含义**：变分推断的下界，作为训练损失函数

**计算公式**：
```python
L = E[log p(z,x,t,y) - log q(z|x,t,y)]
  = 重构损失 - KL散度
```

**包含项**：
- `log p(y|t,z)`: 结果似然
- `log p(t|z)`: 处理似然
- `log p(x|z)`: 协变量似然
- `log p(z) - log q(z|x,t,y)`: KL 散度

**代码位置**：cevae_ihdp.py:132-135

---

## 预测获取方法

### get_y0_y1 函数

**代码位置**：utils.py:30-42

**功能**：从后验预测分布中采样，计算不同处理条件下的预测结果

**参数**：
- `sess`: TensorFlow 会话
- `y`: 后验预测分布
- `f0`: t=0 的 feed_dict
- `f1`: t=1 的 feed_dict
- `L`: 采样次数（训练时用 1，测试时用 100）

**计算过程**：
```python
# 采样 L 次取平均，降低方差
for l in range(L):
    y0 += sess.run(y_post_mean, feed_dict={t=0}) / L
    y1 += sess.run(y_post_mean, feed_dict={t=1}) / L
return y0, y1
```

---

## 指标意义

| 指标 | 越小越好 | 说明 |
|------|:--------:|------|
| ITE | ✓ | 个体层面因果效应估计精度，反映模型对每个样本处理效应的预测能力 |
| ATE | ✓ | 平均因果效应估计精度，反映总体处理效应的估计准确性 |
| PEHE | ✓ | 异质性效应估计精度，是论文主要报告指标 |
| rmse_f | ✓ | 事实结果预测误差，反映模型对已观测结果的拟合能力 |
| rmse_cf | ✓ | 反事实结果预测误差，反映模型对未观测结果的预测能力 |

---

## 参考文献

Louizos et al. (2017). [Causal Effect Inference with Deep Latent-Variable Models](https://arxiv.org/abs/1705.08821). NeurIPS.
