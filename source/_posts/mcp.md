---
title: MCP 协议学习
categories: ai
date: 2025-03-28 11:00:00
typora-root-url: ./mcp
---

# MCP 协议是什么

MCP 是一个开放协议，用于标准化应用程序如何为大语言模型（LLM）提供上下文。可以将 MCP 想象成 AI 应用程序的 USB-C 接口。就像 USB-C 为连接各种外围设备和配件提供了标准化的方式，MCP 为将 AI 模型连接到不同的数据源和工具提供了标准化的方式。

![MCP协议说明](f92b54d519822246291cc942866d4db.png)



# MCP 使用

## 安装

```bash
pip install mcp
pip install mcp[cli]
```



## 运行

python 创建 server.py 文件

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
```



运行 dev

```bash
mcp dev server.py
```



然后在浏览器打开 http://localhost:5173/

能够看到

![MCP Inspector](/image-20250328113411422.png)



## 与 Claude Desktop 联调

执行

```bash
mcp install server.py
```

然后会在 claude 目录下 `AppData\Roaming\Claude` 下生成 `claude_desktop_config.json` 文件，默认是包含 `uv` 命令的，我修改为以下内容：

```json
{
  "mcpServers": {
    "Demo": {
      "command": "mcp",
      "args": [
        "run",
        "D:\\sjj\\script\\mcp_test\\server.py"
      ]
    }
  }
}
```

然后重启 Claude Desktop ，出现了以下内容：

<img src="/image-20250328113652395.png" alt="image-20250328113652395" style="zoom:67%;" />



执行加法：

<img src="/image-20250328113721120.png" alt="image-20250328113721120" style="zoom:67%;" />
