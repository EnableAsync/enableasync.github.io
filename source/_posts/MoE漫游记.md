---
title: MoE漫游记
date: 2026-08-10 01:04:28
tags:
typora-root-url: ./MoE漫游记
---

# MoE 漫游记

> 参考：苏剑林 简单谈谈K3的MoE和Attention https://kexue\.fm/archives/11848
>
> https://kexue\.fm/archives/10735
>

# 为什么使用 MoE，MoE 的缺点

MoE 本质上是对 Transformer 中 FFN 结构的「**近似**」，假设我们有无穷多算力进行 scaling 时，并不需要 MoE，只需要有足够大的 FFN 就能取得很好的效果。

但是在有限的算力情况下，MoE 是一种能够提升 scaling law 曲线斜率的方法（相同算力下取得更好的结果），是因为 MoE 能够在计算量增加较少的情况下更多地增加参数规模。假设有 $n$个专家，激活 $k$个专家，每个专家参数量为 $M$，那么我们近似地用 $kM$ 的参数获得了 $nM$ 参数所该有的智能。

但是 MoE 也存在缺点：多个专家负载不均衡（可能导致退化为只有单个专家处理所有输入，失去优势）、训练不稳定、小 GEMM 效率低 等问题。本文章主要聚焦于解决专家负载不均和训练不稳定 这两个问题。

# MoE 负载均衡

对于 FFN 来说，我们的输出由一个 FFN 层计算得到：

$$
\boldsymbol{y} = \text{FFN}(x)
$$
对于 MoE 来说，输出则由 $k$ 个小型的 FFN 加权混合得到：

$$
\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i
$$
观察公式，其实就是 $k$ 个专家各自的输出 $e_i$ 乘以一个系数 $\rho$ 加权得到最终的输出。于是自然地，MoE 由计算输出的 Export 和计算系数的 Router 构成。

虽然 MoE 公式给人的感觉是「每遇到一个 Token，就去找对应的 Expert 来计算」，但是实际训练的时候是反过来的：先给每个 Export 分配好相应的算力，然后将 Token 分配（Route）到所属的 Export 中并行计算，这也就是为什么负责打分的 $\rho$ 被称为 Router。

这样一来，如果 Export 处理的 Token 的分配不均，就可能出现以下局面：某些 Export（Dead Export）几乎一致闲置，浪费算力；某些 Export 要处理的 Token 太多了，根本忙不过来，只能 Token Drop（放弃处理部分 Token）。从理论上来说，出现 Dead Expert 也意味着 MoE 没有达到预期的参数量，即花了大参数量的显存，结果只训出来小参数量的效果。

所以，不管是从训练还是性能角度看，我们都希望保证 Expert 的负载均衡。

## 辅助损失 Aux Loss

促进负载均衡的常规思路是添加与之相关的损失函数，我们通常称之为“Aux Loss（Auxiliary Loss）”，目前主流用的Aux Loss最早可以追溯到2020年的[《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》](https://papers.cool/arxiv/2006.16668)。

介绍Aux Loss之前，我们需要先引入一些新概念。首先，我们已经提到对于一般的MoE来说，ρ未必是概率分布，我们将归一化的 $\boldsymbol{\rho}$ 记为 $\boldsymbol{p}=[p_1,p_2,\cdots,p_n]$，以及它 Top\-k 版为 $\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$，其中

$$
p_i = \frac{\rho_i}{\sum_{i=1}^n \rho_i},\qquad f_i = \left\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho} \\
0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}\end{aligned}\right.
$$


接着我们定义 $\boldsymbol{P}=\mathbb{E}[\boldsymbol{p}],\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$，这里的 $\mathbb{E}$ 是指对所有样本的所有 Token 做平均。不难看出，$\boldsymbol{F}$ 就是 Expert 当前的负载分布，而 $\boldsymbol{P}$ 则相当于 $\boldsymbol{F}$ 的一个光滑近似。

有了这些记号，我们就可以写出Aux Loss为，推导过程可以见下文：

$$
\mathcal{L}_{\text{aux}} = \boldsymbol{F}\cdot \boldsymbol{P} = \sum_{i=1}^n F_i P_i
$$
一般文献定义 Aux Loss 会多乘一个 $n$，即它们的 Aux Loss 等于这里的 $n \mathcal{L}_{\text{aux}}$。此外，有些大型 MoE 可能会按设备来算Aux Loss，以达到设备内的均衡，减少设备间的通信，这些就各自发挥了。但也有较新的实验显示，强行局部均衡极有可能影响模型最终效果。

## Loss Free 负载均衡 \- Deepseek

Aux Loss固然简单直观，但它也有一个明显的缺点——权重不好调——调低了无法促进均衡，调高了容易损害LM Loss，所以业界一直有寻找替代方案的尝试。这里要分享的是名为“Loss\-Free”的方案，由DeepSeek在[《Auxiliary\-Loss\-Free Load Balancing Strategy for Mixture\-of\-Experts》](https://papers.cool/arxiv/2408.15664)提出。

面对负载不均衡，Aux Loss 的应对思路是通过额外的损失引导 Router 给出均衡的打分，而 Loss\-Free 的想法则是换个新的分配思路，即不改变 Router 现有打分结果，而是改变 $\mathop{\text{argtop}}_k \boldsymbol{\rho}$ 这个分配方式。

其实这个方向此前也有过一些努力。比如2021年Facebook提出了 [BASE Layer](https://papers.cool/arxiv/2103.16716)，将 Expert 的分配视为[线性指派问题](https://en.wikipedia.org/wiki/Assignment_problem)，即以负载均衡为约束条件，求在该约束之下 Router 总打分尽可能高的分配结果，这可以用[匈牙利算法](https://en.wikipedia.org/wiki/Hungarian_algorithm)等来解决。但该方案需要知道全体 Token 的打分，所以对于自回归式 LLM 来说，它只适用于训练，推理还是只能用 $\mathop{\text{argtop}}_k \boldsymbol{\rho}$，训练推理存在不一致性，并且由于目前求解算法的限制，它只适用于 $k=1$ 的场景。

相比之下，Loss\-Free 的做法非常简单且有效，它留意到一个事实，即我们总可以引入一个偏置项 $\boldsymbol{b}$，使得 $\mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}$ 的分配是均衡的，所以它将MoE的形式改为：

$$
\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i
$$
也就是说，之前传统 MoE 选择使用哪些专家由 Router 的权重来确定，我们人为地通过 Aux Loss 给 Router 还增加了一个约束：选出来的专家要均衡。可是有时候这个约束就会损害 Router 本来的功能（选出正确的专家）。

Deepseek 将这个强加给 Router 的负载均衡 Aux Loss 给移除了，用一个可学习的偏置项 $\boldsymbol{b}$ 来专门负责负载均衡，这样就避免了损害模型性能，得到了更好的表现。

这里的 $\boldsymbol{b}$ 是输入无关的向量，由训练过程确定下来，训练完后它就保持不变，因此推理阶段也可以用，换言之训练和推理具有一致的形式。注意乘以 $\boldsymbol{e}_i$ 的还是 $\rho_i$ 而不是 $\rho_i + b_i$，也就是说 $\boldsymbol{b}$ 仅仅参与分配过程而不参与 MoE 的前向计算，所以我们对 $\boldsymbol{b}$ 或 $\boldsymbol{\rho} + \boldsymbol{b}$ 的正负性都没有特殊要求。

这里也给出优化 $\boldsymbol{b}$ 的 Loss，详细推导可以见下文：

$$
\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})
$$
除了加 $\operatorname{sign}$ 的符号梯度下降外，苏剑林发现直接对 $F-Q$ 做 RMS Norm（即 Normalized SGD），在相同的 $\gamma$ 下往往能达到更好的均衡效果：

$$
b \leftarrow b-\gamma\frac{F-Q}{\operatorname{RMS}(F-Q)}
$$


这里的 RMS 是 Root Mean Square ，定义为

$$
\operatorname{RMS}(F-Q)
=
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(F_i-Q_i)^2}
$$
不难看出，加 $\operatorname{sign}$ 后的 $\operatorname{sign}(F-Q)$ 和加 RMS Norm 后的

$$
\frac{F-Q}{\operatorname{RMS}(F-Q)}
$$
它们的 RMS 都是 1，因此它们在尺度上大致相同，所以我们可以使用相同的 $\gamma$。

简单来说，$\operatorname{sign}$ 的问题在于不论 $F_i$ 与目标 $Q_i$ 的远近都使用同样的更新幅度，这导致原本就已经跟 $Q_i$ 比较接近的 $F_i$ 反而容易偏离原本已经达到的均衡，从而产生震荡；而 RMS Norm 则保留了 $F_i-Q_i$ 之间的相对大小，更新幅度更加自适应一些，理论上更有助于促进均衡，实测效果也多是它更好。

有了上面的改进之后，在同一个 step 内，相对接近 Q 的 Export 更新更小；但是还缺一个点：越来越均衡的时候，bias 更新越小，于是我觉得可以加上 Loss 衰减来优化这个点（待做实验）：

$$
b \leftarrow b-\gamma_t\frac{F-Q}{\operatorname{RMS}(F-Q) + \epsilon}
$$


# MoE Scaling

## Latent MoE \- NVIDIA

https://arxiv\.org/abs/2601\.18089

![Image](1786295108792-1.png)

动机和出发点与图像生成中的 Stable Diffusion 类似：在高维空间中需要更多的计算量，因此在低维空间（Latent 空间）中进行操作。

好处是可以在大致相同的训练和推理成本下，实现更好的效果（通过减少参数量，达到相同的效果）。但是也引入了一些稳定性的问题，Kimi 的 Stable LatentMoE 对此进行了解决。

## Stable LatentMoE \- Kimi

当前主流架构的 MoE 都是如下形式：

$$
\boldsymbol{W}_3(\text{SiLU}(\boldsymbol{W}_1 \boldsymbol{x}) \odot \boldsymbol{W}_2 \boldsymbol{x}
$$


其中 $\text{SiLU(x)} = x\sigma(x)$（Sigmoid Linear Unit，亦称 [Swish](https://arxiv.org/abs/1710.05941)），$\sigma$是Sigmoid函数。作为主要非线性来源，SwiGLU经常出现的问题是：$\boldsymbol{W}_1$的某一行$\boldsymbol{w}$与某个输入$\boldsymbol{x}$同向（Align），导致输出$\boldsymbol{w} \cdot {x}$非常大。更极端的是，这种现象在W2x也同时发生，并且发生的位置还一样，于是中间部分出现了 $\mathcal{O}(\Vert\boldsymbol{x}\Vert^4)$ 级别的异常值。

对此，Kimi 先将 SiLU 换成了 SiTU（Sigmoid Tanh Unit）：

$$
\operatorname{SiTU}(x;\beta)
=
\underbrace{\beta\tanh\left(\frac{x}{\beta}\right)}_{\operatorname{softcap}(x;\beta)}
\cdot \sigma(x)
$$
这样先将门控部分控制在 $(-\beta,\beta)$ 内，其中 $\beta=4$。进一步压测发现，这样还不能完全杜绝膨胀，所以 Kimi 干脆把线性部分也加上了 $\operatorname{softcap}$ 运算，形成了如今的 SiTU\-GLU：

$W_3\left(
\operatorname{SiTU}(W_1x;\beta_1)
\odot
\operatorname{softcap}(W_2x;\beta_2)
\right)$

其中 $\beta_1=4,\beta_2=25$。给 SwiGLU 引入 Clip 操作已经不新鲜，GPT\-OSS、DSV4 就已经引入过 Hard Clip 操作，但 Kimi 团队发现，在同样的界限下，$\operatorname{softcap}$ 往往能起到更好的效果，所以选择了 $\operatorname{softcap}$。



# MoE 推导

## MoE 是 FFN 的近似

![image-20260810011153310](image-20260810011153310.png)

![image-20260810011236531](./image-20260810011236531.png)

![image-20260810011246620](./image-20260810011246620.png)

## MoE 负载均衡 Aux Loss

![image-20260810011253626](./image-20260810011253626.png)

![image-20260810011259752](./image-20260810011259752.png)



## MoE 负载均衡 Loss Free 推导

![image-20260810011306635](./image-20260810011306635.png)

![image-20260810011311524](./image-20260810011311524.png)





