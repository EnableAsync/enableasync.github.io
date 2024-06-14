---
title: tensorflow 中的一些坑
date: 2024-06-14 14:20:00
tags: ai
---

# Tensorflow 安装之后无法使用 GPU，但是 Pytorch 可以

有错误提示：
```pain
tensorflow/core/common_runtime/gpu/gpu_device.cc:2251] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
```

原因是 Tensorflow 无法使用 cudnn 的库，找个安装了 Pytorch 的环境，导出其中的 cudnn 路径：
```bash
export CUDNN_PATH=~/miniconda3/envs/sd/lib/python3.12/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":$LD_LIBRARY_PATH
```

之后测试通过：
```python
import tensorflow as tf
print(tf.test.is_gpu_available())
```
