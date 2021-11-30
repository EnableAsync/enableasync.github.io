---
title: kind
date: 2021-11-30 21:40:27
tags: k8s, kind
---

# 记如何使用 Kind 的一些使用方式

## 1. Kind 安装

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.11.1/kind-linux-amd64
chmod +x ./kind
mv ./kind /usr/bin/kind
```

## 2. 使用 Kind 创建含有两个 Node 的 kubernetes 集群

### 1. 创建配置文件

这里我创建了两个 Node，使用以下配置文件，并将其命名为 `kind.yaml`

```yaml
# a cluster with 1 control-plane nodes and 2 workers
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
```

### 2. 创建集群

```bash
sudo kind create cluster --config kind.yaml
```

这里需要注意的点有：

1. 不要设置集群 name，在我本地，如果设置了 name 会导致 kubeconfig 无法导出。
2. 要使用 sudo，在我本地，如果不使用 sudo 会导致无法创建集群，原因未知。

```bash
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.21.1) 🖼
 ✓ Preparing nodes 📦 📦 📦  
 ✓ Writing configuration 📜 
 ✓ Starting control-plane 🕹️ 
 ✓ Installing CNI 🔌 
 ✓ Installing StorageClass 💾 
 ✓ Joining worker nodes 🚜 
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind
```

如果出现以上信息表示创建成功，可以进行下一步。

### 3. 导出 kubeconfig

```shell
kind export kubeconfig
```

如果不使用这一步，会导致使用 `kubectl` 的时候必须加上 `sudo`，否则无法连接到 kubernetes。

## 3. 安装 kubernetes-dashboard

### 1. 使用 helm 安装 dashboard

```bash
# Add kubernetes-dashboard repository
helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
# Deploy a Helm Release named "dashboard" using the kubernetes-dashboard chart
helm install dashboard kubernetes-dashboard/kubernetes-dashboard
```

### 2. 转发 dashboard pod

这一步的目的是在本地访问部署了 dashboard 的 pod

```bash
export POD_NAME=$(kubectl get pods -n default -l "app.kubernetes.io/name=kubernetes-dashboard,app.kubernetes.io/instance=dashboard" -o jsonpath="{.items[0].metadata.name}")
  echo https://127.0.0.1:8443/
  kubectl -n default port-forward $POD_NAME 8443:8443
```

之后会提示

```bash
Forwarding from 127.0.0.1:8443 -> 8443
Forwarding from [::1]:8443 -> 8443
```

说明转发成功，此时访问 https://127.0.0.1:8443/ ，注意是 https

### 3. 生成 token

不出意外 dashboard 需要 token 来登录，使用以下步骤来生成 token：

```bash
kubectl create serviceaccount dashboard -n default
kubectl create rolebinding def-ns-admin --clusterrole=admin --serviceaccount=default:def-ns-admin
kubectl create clusterrolebinding dashboard-cluster-admin --clusterrole=cluster-admin --serviceaccount=default:dashboard
```

```bash
kubectl describe sa dashboard
Name:                dashboard
Namespace:           default
Labels:              <none>
Annotations:         <none>
Image pull secrets:  <none>
Mountable secrets:   dashboard-token-vzzjn
Tokens:              dashboard-token-vzzjn
Events:              <none>
```

这里可以看到 `dashboard-token-vzzjn` 就是我们需要的 token，使用以下命令显示具体内容：

```bash
kubectl describe secret dashboard-token-vzzjn
```

之后就可以将具体的 token 粘贴在 dashboard 中登录。
