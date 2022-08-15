---
title: Linux 的一些使用心得
categories: linux
---

# Linux 下抓 HTTPS 包

## 使用 MITMProxy

1. 运行 MITMProxy

```bash
docker run --rm -it -p 18080:8080 -p 127.0.0.1:8081:8081 -v ~/.mitmproxy:/home/mitmproxy/.mitmproxy  mitmproxy/mitmproxy mitmweb --web-host 0.0.0.0 --set block_global=false --set ssl_insecure=true
```

2. 导入证书至浏览器或其他工具
2. 使用代理访问 HTTPS 页面

# 更新 ubuntu 22.04 之后网易云音乐无法使用

修改 `/opt/netease/netease-cloud-music/netease-cloud-music.bash` 为以下内容
```bash
#!/bin/sh
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}"/libs
export QT_PLUGIN_PATH="${HERE}"/plugins 
export QT_QPA_PLATFORM_PLUGIN_PATH="${HERE}"/plugins/platforms
cd /lib/x86_64-linux-gnu/
exec "${HERE}"/netease-cloud-music $@

```

# 关闭无用启动项

```bash
# 查看启动项
ls -l /etc/xdg/autostart

# 重命名
sudo mv something something.bak
```



# Vmware 更新内核失败

```bash
git clone https://github.com/mkubecek/vmware-host-modules.git
git checkout <your_version>
sudo make
sudo make install
```



# 双系统 Windows 更新失败

我这里双系统 Windows 更新失败的原因是 Windows 引导出现了问题，可以进入 Windows 输入 `msconfig` 查看引导选项卡下是否有内容，我是用过 systemd boot 来引导的 Windows，所以没有出现内容。

在 BIOS 中更改成直接引导 Windows 之后便可以正常更新了。

# 按时间降序最近安装的程序

```bash
for x in $(ls -1t /var/log/dpkg.log*); do 
      zcat -f $x |tac |grep -e " install " -e " upgrade "; 
done | awk -F ":a" '{print $1 " :a" $2}' |column -t
```

# 常用的一些 gnome extensions

## Unite

最大化时隐藏标题栏

## Clear Top Bar

状态栏变成透明的

## ddterm

按 `F10` 快速启动命令行，再按 `F10` 隐藏，十分方便

## Desktop Icons NG(DING)

在桌面上显示图标

## Lock Keys

可以显示当前大小写状况

## NetSpeed

显示当前网速

## TopIcons Plus（在 gnome 40 之后使用 Ubuntu Appindicators 替代）

在顶部显示图标

## Dash to Dock

在底部智能显示一个 Dock

# 换 MAC 地址

有的时候需要更换 linux 的 ip 地址：

```bash
sudo ifconfig eth0 down
sudo ifconfig wlo1 hw ether 02:42:41:7d:b7:6e
sudo ifconfig wlo1 up
```

这里 `eth0` 是网络 interface，ether 之后的参数就是 MAC 地址

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

# Alt + Tab 时阻止相同应用叠加

在 gnome 设置中，打开 keyboard shortcut，将 `Switch windows` 设置为 `Alt + Tab` ，而不是默认的 `Switch applications`。

参考：https://superuser.com/questions/394376/how-to-prevent-gnome-shells-alttab-from-grouping-windows-from-similar-apps

# fluxion

## 扫描不到热点

```bash
sudo airmon-ng
sudo airmon-ng start fluxwl0
```

```bash
export FLUXIONAirmonNG=1
```

执行上述命令后再运行 fluxion 即可。

## 解除 53 端口被 systemd-resolved 占用

1. 先停用 systemd-resolved 服务

```
systemctl stop systemd-resolved
```

2. 编辑 /etc/systemd/resolved.conf 文件

```
vi /etc/systemd/resolved.conf
```

3. 换下面说明更改，然后按一下“esc”键，再输入“:wq”（不要输入引号），回车保存即可。

```
[Resolve]
DNS=8.8.8.8  #取消注释，增加dns
#FallbackDNS=
#Domains=
#LLMNR=no
#MulticastDNS=no
#DNSSEC=no
#Cache=yes
DNSStubListener=no  #取消注释，把yes改为no
```

4. 最后运行下面命令即可。

```
ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf
```

