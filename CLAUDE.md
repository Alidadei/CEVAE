## 环境配置要求：

配置虚拟环境，环境的名称需要由用户确认。所有配环境产生的文件尽量放在项目目录下的统一文件夹，而不要放到C盘；

本机conda路径在：C:\Users\y\miniconda3\condabin

尽量保证不同机器上环境的可复现性。

## 项目结构维护：

使用不同文件夹来归类不同的文件！保持项目结构整洁清晰！

以下文件夹如果项目目录下没有，则创建！

docs 文件夹：所有用户和agent输出的技术文档和经验总结都放在这里

record 文件夹：放所有的实验结果记录

tests文件夹：所有单独测试或问题验证代码都需要放到这个文件夹



## 实验运行

尽量告诉用户怎么运行实验（用最简单的方式），除非用户同意后台自动运行

不要一次性运行多个占用CPU的实验！

不同数据集指南（不同评价指标或流程需要找用户确认：

To employ CEVAE for other datasets you can just mimic the structure of the IHDP class at `datasets.py`. Do note that you will also have to specify appropriate distributions via Edward for the covariates at `x`, treatments at `t` and outcomes at `y`. For example, poisson for covariates which are counts, or categorical/Bernoulli for discrete outcomes.

The definition of the distribution type for the treatment type and outcome can be easily changed by modifying lines 93, 99 for the generative model and by modifying lines 104 and 109 for the inference model at `cevae_ihdp.py`.

Also note that IHDP, being a synthetic dataset, has both the treated and control conditional means (mu1 and mu0) and the factual and counterfactual outcomes (y and y_cf). These are used in evalution.py to calculate various performance metrics. For a dataset without the counterfactuals you will have to avoid calling these evaluation functions and instead write your own evaluation procedure.

## 运行结果保存

重要代码的运行结果（比如主要的训练程序）：如果仅仅是参数，就写入成一份文档保存到 record文件夹，并以实验配置命名（比如所用的数据集和训练策略或者是时间）；结果保存的模板参照record文件夹下的Results format.xlsx，每个模型保存并输出test的相应指标。

如果有图片，仅有单张图片的处理方式参考文档的处理方法，多张图片的情况需要每次在record下新建文件夹（以实验配置和时间命名）把每次运行输出的多张图放在这个文件夹里面。

模型的训练结果要保存，以保证训练结果的可复现性！

！每次代码的运行和测试方式有变更时，都需要及时更新docs目录下的说明文档！

其余测试或问题验证的程序结果，运行前提问用户是否需要将结果保存到record，以什么样的文件形式，如何命名。

## python CPU 性能优化.

### 一、CPU占用率过高的原因

常见的原因包括：

1. **密集型计算**：某些算法或数据处理任务需要大量计算资源。
2. **循环等待**：程序中存在不必要的循环等待，导致CPU空转。
3. **多线程/多进程滥用**：不合理的多线程或多进程使用，增加了上下文切换的开销。
4. **外部库效率低下**：某些第三方库在Windows系统下的性能表现不佳。

### 二、优化技巧与实践

#### 1. 使用更高效的算法

选择合适的算法是降低CPU占用率的关键。例如，对于数据处理任务，可以考虑使用NumPy等高效的科学计算库，它们底层使用C语言优化，能够显著提升计算效率。

**示例代码**：

```python
import numpy as np

# 使用NumPy进行矩阵乘法
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.dot(a, b)
```

#### 2. 避免不必要的循环等待

在编写代码时，应尽量避免不必要的循环等待。可以使用时间间隔或事件驱动的方式来替代空转循环。

**示例代码**：

```python
import time

# 使用时间间隔替代空转循环
while True:
    # 执行任务
    do_something()
    
    # 休眠1秒
    time.sleep(1)
```

#### 3. 合理使用多线程/多进程

多线程和多进程可以提高程序的并发能力，但滥用会导致CPU占用率飙升。应根据任务的性质选择合适的并发模型。

**示例代码**：

```python
import threading

def task():
    # 执行任务
    pass

# 创建并启动线程
thread = threading.Thread(target=task)
thread.start()
thread.join()
```

#### 4. 优化外部库的使用

某些外部库在Windows系统下的性能可能不佳。可以通过以下方式优化：

- **更新库版本**：使用最新版本的库，通常包含性能优化。
- **替换库**：寻找性能更好的替代库。
- **自定义实现**：对于关键代码段，可以考虑自己实现优化版本。

**示例代码**：

```python
# 使用最新版本的库
pip install --upgrade some_library
```

#### 5. 使用性能分析工具

借助性能分析工具，可以定位CPU占用率高的代码段，进行针对性优化。常用的工具包括cProfile、line_profiler等。

**示例代码**：

```python
import cProfile

def main():
    # 主函数代码
    pass

# 使用cProfile进行性能分析
cProfile.run('main()')
```

### 三、案例分析：优化一个实际项目

假设我们有一个数据处理的Python项目，运行时CPU占用率高达90%以上。通过以下步骤进行优化：

1. **性能分析**：使用cProfile定位到数据处理函数为瓶颈。
2. **算法优化**：将数据处理函数中的循环计算改为使用NumPy矩阵操作。
3. **多线程优化**：将数据处理任务分配到多个线程中并行执行。
4. **结果对比**：优化后，CPU占用率降至30%左右，程序运行速度提升显著。