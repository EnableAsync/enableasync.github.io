---
title: 远程服务器的 Git 使用本地的密钥——ssh-agent
categories: linux
---

# 动机
有的时候我们在远程服务器上开发，但是远程服务器有多个人使用，我们并不想把自己的私钥放到服务器上，但是又想通过私钥连接到 Github 等远程仓库，那么该怎么做呢？

# SSH Agent
SSH Agent 是一个程序，用于管理多个 SSH 密钥并为 SSH 协议提供身份验证服务。它主要用于以下几点：
- 密钥管理：SSH Agent 可以存储用户的私钥，并在需要时提供给 SSH 客户端，从而避免每次连接时都需要输入密码。
- 身份验证：当用户尝试连接到远程服务器时，SSH Agent 会自动处理身份验证过程，提高安全性。
- 多会话支持：用户可以在多个终端会话或应用程序中使用同一个 SSH Agent，而不需要重复加载密钥。
- 
在使用 SSH 进行远程操作时，启动 SSH Agent 并将私钥添加到其中，可以简化连接过程并提高效率。

例如，在 Linux 系统中，可以通过以下命令启动 SSH Agent 并添加私钥：
```bash
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa
```
这样，后续的 SSH 连接将自动使用这些密钥进行身份验证。

# 在 Windows 上开启 SSH Agent
根据微软帮助，我们能够通过以下方式开启 ssh-agent 并加入我们的私钥：
```bash
# By default the ssh-agent service is disabled. Configure it to start automatically.
# Make sure you're running as an Administrator.
Get-Service ssh-agent | Set-Service -StartupType Automatic

# Start the service
Start-Service ssh-agent

# This should return a status of Running
Get-Service ssh-agent

# Now load your key files into ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ecdsa

# Check that it worked
ssh-add -l
```

# vscode 中启用 SSH Agent
首先要在 ssh 中开启 `ForwardAgent`，具体方法是编辑 ssh config：
```
Host async-004
    HostName IP
    ForwardAgent yes
```
之后在本地的 vsocde 设置中搜索 `Enable Agent Forwarding` 并开启。

之后 F1 搜索 Kill，关闭远程 vscode server，之后重新连接远程服务器，这个时候远程服务器上输入 `ssh-add -l` 应该能看到我们之前在 Windows 上添加的私钥，就实现了密钥的安全转发。

# 参考

1. https://learn.microsoft.com/zh-cn/windows-server/administration/openssh/openssh_keymanagement
2. https://zhuanlan.zhihu.com/p/394555196
