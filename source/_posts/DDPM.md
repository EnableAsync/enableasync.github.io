---
title: 算法整理
tags: data structure, algorithm
mathjax: true
date: 2025-04-3 14:23:00
typora-root-url: ./DDPM
---

# DDPM

![ddpm](image-20250403144618848.png)

这里从 $x_t$ 到 $x_{t-1}$ 的时候，预测的是 $x_{t-1}$ 的概率分布，也就是说要得到的是 $p(x_{t-1}|x_t)$，所以去噪过程是具有一定的随机性的，从 $p(x_{t-1}|x_t)$ 分布中就可以抽样出 $x_{t-1}$ 了。

![笔记](53dfabcb9844f43a33026fe042dff8c.jpg)

## 去噪过程

$$
P(x_{t - 1}|x_t) = \frac{P(x_{t - 1}, x_t)}{P(x_t)} = \frac{P(x_t|x_{t - 1})P(x_{t - 1})}{P(x_t)} \quad \text{(1)}
$$
这里的 $$P(x_t|x_{t - 1})$$ 是易知的，因为 $$x_t = \sqrt{\alpha_t}x_{t - 1} + \sqrt{\beta_t}\epsilon_t$$，其中 $$\epsilon_t \sim N(0, 1)$$，$$\sqrt{\beta_t} \sim N(0, \beta_t)$$，$$x_t \sim N(\sqrt{\alpha_t}x_{t - 1}, \beta_t)$$。

$$P(x_t|x_{t - 1}) \sim N(\sqrt{\alpha_t}x_{t - 1}, \beta_t)$$ 不断推导有 $$P(x_t|x_0) \sim N(\sqrt{\bar{\alpha}_t}x_0, 1 - \bar{\alpha}_t)$$。

- 原先的去噪过程中 $$P(x_{t - 1})$$ 和 $$P(x_t)$$ 无法求得，为去噪过程增加条件：
  $$
  P(x_{t - 1}|x_t, x_0) = \frac{P(x_t|x_{t - 1}, x_0)P(x_{t - 1}|x_0)}{P(x_t|x_0)} = \frac{P(x_t|x_{t - 1})P(x_{t - 1}|x_0)}{P(x_t|x_0)}
  $$
  

  （高斯特性）又 $$P(x_t|x_0)$$ 已知，则 $$P(x_{t - 1}|x_0)$$ 已知，得 $$P(x_{t - 1}|x_t, x_0) \sim N(\tilde{\mu}_t(x_0, x_t), \tilde{\beta}_t)$$。

  $$x_{t - 1} = \tilde{\mu}_t(x_0, x_t) + \sqrt{\tilde{\beta}_t}\epsilon$$，$$\epsilon \sim N(0, 1)$$，其中 $$\tilde{\mu}_t(x_t, x_0)$$ 为 $$\frac{\sqrt{\alpha_{t - 1}}\beta_t}{1 - \bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}x_t$$  ，$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}\beta_t$$ 。这里的 $$x_0$$ 未知。

  

- 于是又有 $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$ ，则 $$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)$$ ，$$x_0$$ 可由 $$x_t$$ 得到了，它们之间差一个 $$\epsilon$$ 。

- UNet 预测的就是 $$x_0$$ 和 $$x_t$$ 之间的这个噪声。那么能得到 $$x_0$$ ，是不是原因呢？否定的，因为 $$x_0$$ 的推导是借助了马尔可夫性质，所以还是要一步步推导。

  把 $$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)$$ 代入 $$\tilde{\mu}_t(x_t, x_0)$$ 式中，有 $$\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon)$$ 。

- 总体脉络：从 $$P(x_{t - 1}|x_t) \to P(x_{t - 1}|x_t, x_0) \to \tilde{\mu}_t(x_0, x_t) \to x_0 \to \epsilon$$ 



