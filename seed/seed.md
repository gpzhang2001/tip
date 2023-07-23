随机种子用于提升程序的可复现性。

## 背景

训练代码过程中，由于存在很多随机值，这些随机值很容易影响网络的训练结果。

常见的有：
1、随机权重，网络有些部分的权重没有预训练，它的值则是随机初始化的，每次随机初始化不同会导致结果不同。

2、随机数据增强，一般来讲网络训练会进行数据增强，特别是少量数据的情况下，数据增强一般会随机变化光照、对比度、扭曲等，也会导致结果不同。

3、随机数据读取，喂入训练数据的顺序也会影响结果。
（实际还存在一些其他的随机值，但是这三个是比较常见或者说比较主要的）

## 解决方法

### 什么是随机种子

随机种子（Random Seed）是计算机专业术语。一般计算机的随机数都是伪随机数，以一个真随机数（种子）作为初始条件，然后用一定的算法不停迭代产生随机数。
所以建党来说，由于这还是一个伪随机数，因此实际算法是固定的，只要随机种子一确定，后续生成的序列将是固定的。

举个例子🌰：
```python
import random

# 设置随机种子
random.seed(11)

# 生成随机整数
print("随机生成")
print(random.randint(1,100))
print(random.randint(1,100))

# 重置随机数生成器种子
random.seed(11)

# 生成随机整数
print("再次随机生成")
print(random.randint(1,100))
print(random.randint(1,100))

# result
随机生成
58
72
再次随机生成
58
72
```

不难看出，只要随机种子一确定，后续序列就是一样的了。

因此在训练过程中，凡是涉及到随机数的地方，均设置好随机种子，那么任何一次训练中，由随机种子生成的随机序列都会是一致的。

### 训练中如何使用

实际训练过程中，有多个库会涉及到随机数的问题。
1、torch库；
 2、numpy库；
 3、random库。

因此，在开启训练之前，需要对这些库设置随机种子。

```python
def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch官方给予的建议
    torch.backends.cudnn.deterministic = True # 用于保证CUDA 卷积运算的结果确定。
    torch.backends.cudnn.benchmark = False # 是用于保证数据变化的情况下，减少网络效率的变化。为True的话容易降低网络效率。
```

除此之外，Pytorch一般使用Dataloader来加载数据，Dataloader一般会使用多worker加载多进程来加载数据，此时我们需要使用Dataloader自带的worker_init_fn函数初始化Dataloader启动的多进程，这样才能保证多进程数据加载时数据的确定性。

```python
def worker_init_fn(worker_id, rank, seed):
  	# rank是分布式训练时需要的
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```





