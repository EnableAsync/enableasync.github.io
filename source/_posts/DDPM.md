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




# Classifier Guidance / Classifier - Free Guidance

## 内容说明
1. **①**：在分类器引导（CG）之前，FID不如 GAN（生成对抗网络）。  
2. **②**：扩散模型（diffusion model）原本控制性不佳，条件生成可提升 FID。CG 本质是一种采样方法，可利用训练好的 DDPM（Denoising Diffusion Probabilistic Model）。  

### 公式推导
- $\hat{q}(x_{t-1} | x_t, y) = \frac{ \hat{q}(x_{t-1} | x_t) \hat{q}(y | x_{t-1}, x_t) }{ \hat{q}(y | x_t) }$ 
  注：$\hat{q}(y | x_t)$ 为常数，与 $x_{t-1}$ 无关。 
- 定义： 
  $\hat{q}(x_t | x_{t-1}, y) := q(x_t | x_{t-1})$，$\hat{q}(x_0) := q(x_0)$，无需重新训练，与 DDPM 相同。 
- $\nabla \hat{q}(x_{t-1} | x_0, y) = \prod_{t'=1}^{t} \hat{q}(x_{t'} | x_{t'-1}, y)$（链式展开）。 

### 详细推导步骤
1. **①**：
   $\hat{q}(x_{t-1} | x_t) = \frac{ \hat{q}(x_t | x_{t-1}) \hat{q}(x_{t-1}) }{ \hat{q}(x_t) }$（贝叶斯公式应用）。
2. **②**： 
   - 推导 $\hat{q}(x_t | x_{t-1})$：
     $\hat{q}(x_t | x_{t-1}) = \int_y \hat{q}(x_t, y | x_{t-1}) dy$ 
     $= \int_y \hat{q}(x_t | y, x_{t-1}) \hat{q}(y | x_{t-1}) dy$ 
     $= \int_y q(x_t | x_{t-1}) \hat{q}(y | x_{t-1}) dy$（因 $\hat{q}(x_t | y, x_{t-1}) = q(x_t | x_{t-1})$） 
     $= q(x_t | x_{t-1}) \int_y \hat{q}(y | x_{t-1}) dy$ 
     $= q(x_t | x_{t-1})$（因 $\int_y \hat{q}(y | x_{t-1}) dy = 1$）。
   - 推导 $\hat{q}(x_t)$： 
     $\hat{q}(x_t) = \int_{x_{0:t-1}} \hat{q}(x_{0:t}) d x_{0:t-1}$ 
     $= \int_{x_{0:t-1}} \hat{q}(x_0) \hat{q}(x_{1:t} | x_0) d x_{0:t-1}$。
     进一步展开 $\hat{q}(x_{1:t} | x_0)$：
     $\hat{q}(x_{1:t} | x_0) = \int_y \hat{q}(x_{1:t}, y | x_0) dy$ 
     $= \int_y \hat{q}(x_{1:t} | y, x_0) \hat{q}(y | x_0) dy$
     $= \int_y \hat{q}(y | x_0) \prod_{t'=1}^t \hat{q}(x_{t'} | x_{t'-1}, y) dy$
     $= \int_y \hat{q}(y | x_0) \prod_{t'=1}^t q(x_{t'} | x_{t'-1}) dy$（因 $\hat{q}(x_{t'} | x_{t'-1}, y) = q(x_{t'} | x_{t'-1})$）
     $= \int_y \hat{q}(y | x_0) q(x_{1:t} | x_0) dy$。





$\hat{q}(x_t)$ 的推导
$$
\hat{q}(x_t) = \int_{x_{0:t-1}} q(x_0) q(x_{1:t} | x_0) dx_{0:t-1} = \int_{x_{0:t-1}} q(x_{0:t}) dx_{0:t-1} = q(x_t)
$$
**说明**：通过积分运算，证明了 $\hat{q}(x_t)$ 与 $q(x_t)$ 相等，体现了分布在积分变换下的不变性。

$q(y | x_t, x_{t-1})$ 的化简
$$
q(y | x_t, x_{t-1}) = \frac{ \hat{q}(x_0 | y, x_{t-1}) \hat{q}(y | x_{t-1}) }{ \hat{q}(x_0 | x_{t-1}) } = \frac{ \hat{q}(x_0 | y, x_{t-1}) }{ \hat{q}(x_0 | x_{t-1}) } \hat{q}(y | x_{t-1})
$$
进一步化简：
$$
= q(x_t | x_{t-1}) \frac{ \hat{q}(y | x_{t-1}) }{ \hat{q}(x_0 | x_{t-1}) }
$$
**说明**：利用条件概率公式进行变形，结合已知分布关系化简，展示条件概率与其他分布的关联。

$\hat{q}(x_{t-1} | x_t, y)$ 的表达式
$$
\hat{q}(x_{t-1} | x_t, y) = \frac{ q(x_{t-1} | x_t) \hat{q}(y | x_{t-1}) }{... }
$$
**标注说明**：红色标注强调与 **DDPM 模型** 的联系，表明该公式在 DDPM 框架下的应用特性。

### 采样方式与近似推导
- **采样式**：$x_t = \mu + \epsilon$，其中 $\epsilon$ 很小。
- **对数概率展开**：
$$
  \log P_\phi(y | x_t) \approx \text{泰勒展开近似}
$$
  进一步对 $\log P_\phi(x_{t-1} | x_t, y)$ 推导：
$$
  \log P_\phi(x_{t-1} | x_t, y) = -\frac{1}{2}(x_t - \mu)^T \Sigma^{-1}(x_t - \mu) + (x_t - \mu) \nabla + C
$$
  通过变形：
$$
  = -\frac{1}{2}(x_t - \mu - \Sigma \nabla)^T \Sigma^{-1}(x_t - \mu - \Sigma \nabla) + C'
$$
  近似为正态分布：
$$
\sim N(\mu + \Sigma \nabla, \Sigma^2) \implies x_t = \mu + \Sigma \nabla + \Sigma \epsilon
$$
**说明**：描述了采样的形式，通过对数概率的展开和变形，推导得出近似正态分布的结果，展示了从概率表达式到采样公式的推导过程。

通过上述推导，展现了分类器引导（CG）相关的采样方法与公式逻辑，基于 DDPM 框架且无需重新训练，体现了其在扩散模型中的应用特性。 



![5d117d10d332d1d5f747467b99e798f](5d117d10d332d1d5f747467b99e798f.jpg)



![c63015d4464ad72db9343356fc0a66c](c63015d4464ad72db9343356fc0a66c.jpg)



# SDE（随机微分方程）下的 Diffusion Model

## 随机过程基础
- 布朗运动增量：$W(t+\Delta t) - W(t) \sim N(0, \Delta t)$，当 $\Delta t \to 0$ 时，可表示为微分形式 $dW = \sqrt{dt} \epsilon$，其中 $\epsilon \sim N(0,1)$，即 $dW \sim N(0, dt)$。
- Itô 过程（扩散过程）：$dX = f(x,t)dt + g(t)dW$，描述了系统状态 $X$ 随时间 $t$ 的变化，包含确定性项 $f(x,t)dt$ 和随机性项 $g(t)dW$。

## 模型与 SDE 关联的优势
1. **数学方法紧密结合**：SDE 提供了丰富的数学工具，便于分析和求解。
2. **刻画分布转换**：能更好地描述数据分布（data distribution）与先验分布（prior distribution）之间的相互转换，因为扩散（加噪）的逆向过程同样是扩散过程。

## 逆向过程公式
- 一般形式：$dX = [f(x,t) - g^2(t) \nabla_x \log p_t(x)]dt + g_t d\tilde{W}$。
- 在 DDPM 中：
  - 正向扩散：$dX = -\frac{1}{2} \beta(t) X(t)dt + \sqrt{\beta(t)} dW$。
  - 逆向过程：$dX = [-\frac{1}{2} \beta(t) X(t) - \beta(t) S_\theta(t)]dt + \sqrt{\beta(t)} dW$，其中 $S_\theta(t) = \nabla_{x_t} \log P(x_t | x_0) = -\frac{x_t - \mu_t}{\sigma^2}$（通过对 $P(x_t | x_0)$ 求梯度得到）。
  - 由 $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$ 代入化简可得 $S_\theta(t) = -\frac{\epsilon}{\sqrt{1 - \alpha_t}}$。

## 数值解法（欧拉方法）
- **SDE 欧拉近似**：$X(t+\Delta t) = X(t) + f(x,t)\Delta t + g(t)\Delta W$。
- **ODE 欧拉法**：$\frac{dx}{dt} = a(X(t))$，则 $X(t+\Delta t) = X(t) + a(X(t))\Delta t$。
- 逆向扩散采样（reverse diffusion sampler）的数值估计：
  - $x_{i+1} = x_i + f(x,t) + g(t)\epsilon$（简化形式，$\Delta t$ 融入 $f$ 和 $g$）。
  - 进一步推导 $x_i$ 的表达式：$x_i = x_{i+1} + \frac{1}{2} \beta_{i+1} x_{i+1} + \beta_{i+1} S_\theta + \sqrt{\beta_{i+1}} \epsilon$，通过近似（如泰勒公式 $(1+x)^\alpha \approx 1+\alpha x$）化简，最终得出 DDPM 是欧拉方法的特例，如 $(2 - \sqrt{1 - \beta(t)})x_{t'} + \beta'(t) S_\theta + \sqrt{\beta'(t)} \epsilon$。

以上内容系统阐述了 SDE 框架下扩散模型的数学基础、逆向过程及数值解法，体现了 SDE 在扩散模型分析中的重要作用。 



![b012c3a57ba9325b7e6acd7fd6703d0](b012c3a57ba9325b7e6acd7fd6703d0.jpg)
