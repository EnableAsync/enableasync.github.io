---
title: Linux 中的一些坑
categories: linux
---

# 输入法

## Fcitx 失效

1. 使用 im-config 修复

2. 可能是 fcitx 没有正常启动，即还是 ibus，可以修改 ~/.pam_environment

3. 删除 /etc/profile.d/pop-im-ibus.sh
   
   `/etc/profile.d/pop-im-ibus.sh` （源文件： /etc/gdm3/Xsession ）设置了环境变量 `XMODIFIERS` ，在 `/etc/X11/Xsession.d/70im-config_launch` 中有如下代码：
   
   ```bash
   if [ -z "$XMODIFIERS" ] && \  # 如果环境变量 XMODIFIERS 没有被设置
      ...
      # 设置环境变量以启动用户指定的输入法
   fi
   ```
   
   因为 `XMODIFIERS` 被设置了，所以 `设置环境变量以启动用户指定的输入法` 没有执行，所以 fcitx 没有被启动。
   
   `/etc/profile.d/pop-im-ibus.sh` 第一次出现于 `pop-os_20.10_amd64_intel_4.iso` （发布于 2020 年 12 月中旬）
   
   相关 issue，https://github.com/pop-os/pop/issues/1445
# Dash to dock
## Dash to dock 重叠问题

   Pop os 自带的 Dock 与 Dash to dock 发生了重叠

   ```shell
   cd /usr/share/gnome-shell/extensions
   sudo mv cosmic-dock@system76.com cosmic-dock@system76.com.bak # 关闭自带的 dock
   ```

   之后重启 gnome 即可解决

