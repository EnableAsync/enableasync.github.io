---
title: Diffusion 学习
tags: data structure, algorithm
mathjax: true
date: 2025-04-3 14:23:00
typora-root-url: ./DDPM
---

# DDPM

![ddpm](image-20250403144618848.png)

这里从 $x_t$ 到 $x_{t-1}$ 的时候，预测的是 $x_{t-1}$ 的概率分布，也就是说要得到的是 $p(x_{t-1}|x_t)$，所以去噪过程是具有一定的随机性的，从 $p(x_{t-1}|x_t)$ 分布中就可以抽样出 $x_{t-1}$ 了。

## 基本方法


$$
P(x_{t - 1}|x_t) = \frac{P(x_{t - 1}, x_t)}{P(x_t)} = \frac{P(x_t|x_{t - 1})P(x_{t - 1})}{P(x_t)} \quad \text{(1)}
$$
这里的 $P(x_t|x_{t - 1})$ 是易知的，因为 $x_t = \sqrt{\alpha_t}x_{t - 1} + \sqrt{\beta_t}\epsilon_t$，其中 $\epsilon_t \sim N(0, 1)$，$\sqrt{\beta_t} \sim N(0, \beta_t)$，$x_t \sim N(\sqrt{\alpha_t}x_{t - 1}, \beta_t)$。

$P(x_t|x_{t - 1}) \sim N(\sqrt{\alpha_t}x_{t - 1}, \beta_t)$ 不断推导有 $P(x_t|x_0) \sim N(\sqrt{\bar{\alpha}_t}x_0, 1 - \bar{\alpha}_t)$。



- 原先的去噪过程中 $P(x_{t - 1})$ 和 $P(x_t)$ 无法求得，为去噪过程增加条件：
  $$
  P(x_{t - 1}|x_t, x_0) = \frac{P(x_t|x_{t - 1}, x_0)P(x_{t - 1}|x_0)}{P(x_t|x_0)} = \frac{P(x_t|x_{t - 1})P(x_{t - 1}|x_0)}{P(x_t|x_0)}
  $$
  

（高斯特性）又 $P(x_t|x_0)$ 已知，则 $P(x_{t - 1}|x_0)$ 已知，得 $P(x_{t - 1}|x_t, x_0) \sim N(\tilde{\mu}_t(x_0, x_t), \tilde{\beta}_t)$。

$x_{t - 1} = \tilde{\mu}_t(x_0, x_t) + \sqrt{\tilde{\beta}_t}\epsilon$，$\epsilon \sim N(0, 1)$，其中 $\tilde{\mu}_t(x_t, x_0)$ 为 $\frac{\sqrt{\alpha_{t - 1}}\beta_t}{1 - \bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}x_t$  ，$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}\beta_t$ 。这里的 $x_0$ 未知。

- 于是又有 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ ，则 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)$ ，$x_0$ 可由 $x_t$ 得到了，它们之间差一个 $\epsilon$ 。

- UNet 预测的就是 $x_0$ 和 $x_t$ 之间的这个噪声。那么能得到 $x_0$ ，是不是原因呢？否定的，因为 $x_0$ 的推导是借助了马尔可夫性质，所以还是要一步步推导。

  把 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)$ 代入 $\tilde{\mu}_t(x_t, x_0)$ 式中，有 $\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon)$ 。

- 总体脉络：从 $P(x_{t - 1}|x_t) \to P(x_{t - 1}|x_t, x_0) \to \tilde{\mu}_t(x_0, x_t) \to x_0 \to \epsilon$



![笔记](53dfabcb9844f43a33026fe042dff8c.jpg)



## 参数选择

1. $ T = 1000 $
2. $ x_t=\sqrt{\alpha_t}x_{t - 1}+\sqrt{\beta_t}\epsilon $，$ \alpha_t<1 $，$ \beta_t>0 $，$ \alpha_t+\beta_t = 1 $
   - $ x_t\sim N(\sqrt{\alpha_t}x_{t - 1},\beta_t) $
   - $ \alpha_t+\beta_t = 1 $ 的设置是为了最后易于得到 $ x_n=\sqrt{\overline{\alpha}_n}x_0+\sqrt{1 - \overline{\alpha}_n}\epsilon $ 的形式
3. **能否直接跳步？**
   - $ P(x_{t - 1}|x_t,x_0)=\frac{P(x_t|x_{t - 1})P(x_{t - 1}|x_0)}{P(x_t|x_0)} $ 是由贝叶斯定理得到的
   - 结论：不能跳步
4. **为什么不直接预测 $ x_{t - 1} $，而是预测 $ P(x_{t - 1}|x_t) $**
   - 这样有更多的多样性
5. **论文中的变分下界分析**
   - 是为了证明 $ E(-\log P(x_0)) $ 等价于求 $ P(x_{t - 1}|x_t) $ 和 $ P(x_{t - 1}|x_t,x_0) $ 的 KL 散度。



![参数选择](34ff3b1a482f1445724b9f1683bbd71.jpg)



# DDIM

## 去马尔可夫化

$$
P(x_{t-1} | x_t) = \frac{P(x_t | x_{t-1})P(x_{t-1})}{P(x_t)}
$$
在 DDPM 不好求，加入 $x_0$
$$
P(x_{t-1} | x_t, x_0) = \frac{P(x_t | x_{t-1}, x_0)P(x_{t-1} | x_0)}{P(x_t | x_0)} \overset{\text{DDPM}}{\Longrightarrow} \frac{P(x_t | x_{t-1})P(x_{t-1} | x_0)}{P(x_t|x_0)}
$$
$$
\overset{\text{DDIM}}{\Longrightarrow} P(x_s | x_k, x_0) = \frac{\text{①}P(x_k | x_s, x_0) \text{②}P(x_s | x_0)}{\text{③}P(x_k | x_0)} \quad (1)
$$

- ① ② ③ 这三部分共同决定了 $P(x_s | x_k, x_0)$ 的解。
- 3 个部分未知的越多，那么说明 $P(x_s | x_k, x_0)$ 的约束越少，解越多。

但 DDPM 训练时，要满足 $P(x_t | x_0) = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$。若不重新训练，那么这个假设仍然要保留，于是 ② ③ 是知道的。

在 DDPM 中，① 并没有被用到，则 (1) 式的解可以设为：
$$
P(x_s | x_k, x_0) \sim \mathcal{N}(kx_0 + mx_k, \sigma^2 I)
$$
于是
$$
\begin{align*}
x_s &= (kx_0 + mx_k) + \sigma \epsilon \\
&= kx_0 + m(\sqrt{\bar\alpha_k}x_0 + \sqrt{1 - \bar\alpha_k}\epsilon') + \sigma \epsilon \\
&= (k + m\sqrt{\bar\alpha_k})x_0 + (m\sqrt{1 - \bar\alpha_k}\epsilon' + \sigma \epsilon) \\
&= (k + m\sqrt{\bar\alpha_k})x_0 + \sqrt{m^2(1 - \bar\alpha_k) + \sigma^2}\epsilon \quad \mathcal{N}(0, m^2(1 - \bar\alpha_k))\\
&= \sqrt{\bar\alpha_s}x_0 + \sqrt{1 - \bar\alpha_s}\epsilon
\end{align*}
$$

**待定系数法求出 $m$ 和 $k$**
$$
m = \frac{\sqrt{1 - \bar\alpha_s + \sigma^2}}{\sqrt{1 - \bar\alpha_k}}, \quad k = \sqrt{\bar\alpha_s} - \frac{\sqrt{1 - \bar\alpha_s - \sigma^2}}{\sqrt{1 - \bar\alpha_k}}\sqrt{\bar\alpha_k}
$$
于是有
$$
\mu = mx_k + kx_0 = \sqrt{\bar\alpha_s}x_0 + \frac{\sqrt{1 - \bar\alpha_s - \sigma^2}}{\sqrt{1 - \bar\alpha_k}}(x_k - \sqrt{\bar\alpha_k}x_0)
$$
以及 $\sigma = \sigma$

相比于 DDPM 的 $s = k - 1$，DDIM 的 $s \leq k - 1$，于是可以跳步了。 



![去马尔可夫化](9ec363832d7635f22955f09c2c27248.jpg)



## 参数选择

- $\sigma = 0$ 时为决定性采样方式，无随机性。
- $\sigma = \sqrt{\frac{1 - \overline{\alpha}_t}{1 - \overline{\alpha}_t} \beta_t}$ 时对应 DDPM 。

$$
P(x_s | x_k, x_0) \sim \mathcal{N}(\sqrt{\alpha_s}x_0 + \sqrt{1 - \alpha_s - \sigma^2} \frac{x_k - \sqrt{\alpha_k}x_0}{\sqrt{1 - \alpha_k}}, \sigma^2 I)
$$
当 $\sigma$ 取 $\sqrt{\frac{1 - \overline{\alpha}_t}{1 - \overline{\alpha}_t} \beta_t}$ 时，原式为 $\mathcal{N}(\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \overline{\alpha}_t}}\epsilon), \beta_t)$ 。

### **DDIM 实验设置**

<img src="image-20250403231724755.png" alt="DDIM 实验结果" style="zoom:80%;" />

$\sigma = \eta \sqrt{\frac{1 - \overline{\alpha}_t}{1 - \overline{\alpha}_t} \beta_t}$

- $\eta = 1$ 时即为可跳步的 DDPM 。
- $\eta = 0$ 时，为确定性 DDIM 。
- $\eta$ 越小，效果越好（FID 越低）。

**总结**：
- DDPM：高斯去噪过程，推理慢。
- DDIM：去噪，推理快，待定系数法求 $m$，$k$ 。以及 $\sigma = 0$ 时，对于每一个 $x_t$ 有唯一确定的 $x$ 。 





![参数选择](34ff3b1a482f1445724b9f1683bbd71.jpg)



