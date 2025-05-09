---
title: Jetbrains 使用指南
date: 2025-05-09 17:00:00
categories: linux
typora-root-url: ./jetbrains
---

# 动机
Jetbrains 会需要我们激活，怎么办呢？

# 步骤

## 卸载 Jetbrains

**注意要卸载干净**

![卸载老版本的 IDEA](172264713794725.jpeg)



## 安装新版



![IDEA 2025.1 安装步骤](174502512804671.jpeg)

## 激活步骤

运行 jetbra 中的 `scripts/install-all-users.vbs`

**记得地区不要选择中国。**

jetbra/ja-netfaliter激活的原理是拦截并重定向与Jetbrains账号验证服务器的数据。
2024.2后jetbrains新的安装程序自带了三个区域语言包，其中若选择中文大陆区域语言包，会将激活验证服务器地址修改为国内新验证地址。而jetbra/ja-netfaliter的拦截是黑名单制度，该新服务器地址并不在原本的拦截列表中。

如果要拦截中国的话，修改jetbra/ja-netfaliter的配置文件，在**URL**配置文件中添加：

```
PREFIX,https://account.jetbrains.com.cn/lservice/rpc/validateKey.action
```

