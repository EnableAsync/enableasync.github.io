---
title: Docker 镜像都失效之后的处理
date: 2025-02-12 19:21:50
categories: linux
---

# Docker 镜像都失效之后的处理

在 2024 年，所有的 docker 镜像已经失效了，想要继续进行 `docker pull` 就需要自建镜像，或者使用代理进行 pull，这里记录一下具体的处理方式。

## Docker system 代理

**在执行docker pull时，是由守护进程dockerd来执行。因此，代理需要配在dockerd的环境中。而这个环境，则是受systemd所管控，因此实际是systemd的配置。**

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo touch /etc/systemd/system/docker.service.d/proxy.conf
```

在这个proxy.conf文件（可以是任意\*.conf的形式）中，添加以下内容：

```bash
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:10800"
Environment="HTTPS_PROXY=http://127.0.0.1:10800"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

注意，当使用 k8s 的时候，容器的 `http://127.0.0.1:10800` 是不可访问的，需要设置为 `docker0` 的 ip 地址。

## Docker 容器内部代理

**在容器运行阶段，如果需要代理上网，则需要配置 ~/.docker/config.json。以下配置，只在Docker 17.07及以上版本生效。**

```bash
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://127.0.0.1:10800",
     "httpsProxy": "http://127.0.0.1:10800",
     "noProxy": "localhost,127.0.0.1,.example.com"
   }
 }
}
```



# SSH 端口转发

## 本地访问远程端口
```bash
ssh -L [LOCAL_IP:]LOCAL_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
```

## 远程访问本地的端口
```bash
ssh -R [REMOTE:]REMOTE_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
```

比如要把本机的代理 `http://127.0.0.1:10800` 端口共享到远程的所有 IP 上的 10801 端口，则是

```bash
ssh -R 0.0.0.0:10801:127.0.0.1:10800 username@ip -p port
```

注意，默认是无法在远程服务器上监听 `0.0.0.0` 的，如果想要监听，需要修改 `/etc/ssh/sshd_config` 中的 `GatewayPorts yes` 才行。

