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

![MCP Inspector](image-20250328113411422.png)



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

<img src="image-20250328113652395.png" alt="image-20250328113652395" style="zoom:67%;" />



执行加法：

<img src="image-20250328113721120.png" alt="image-20250328113721120" style="zoom:67%;" />



# 自己编写一个操作数据库的 MCP Server

## server.py 代码

```python
import MySQLdb
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MySQL Explorer")

conn = MySQLdb.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="root",
)

@mcp.resource("schema://main")
def get_schema() -> str:
    """Provide the database schema as a resource"""
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")  # Get a list of all tables

    tables = cursor.fetchall()
    schema = []

    for table in tables:
        table_name = table[0]
        cursor.execute(f"SHOW CREATE TABLE `{table_name}`")  # Get the create statement for each table
        create_table_sql = cursor.fetchone()[1]
        schema.append(create_table_sql)
    return "\n".join(schema)



@mcp.tool()
def query_data(sql: str) -> str:
    """Execute SQL queries safely"""
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return "\n".join(str(row) for row in result)
    except Exception as e:
        return f"Error: {str(e)}"
    

```



## 然后 MCP Inspector 获取 resource

```bash
curl 'http://localhost:3000/message?sessionId=c73d174d-8772-441a-bfa1-1771ad358aa1' \
  -H 'Accept: */*' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6' \
  -H 'Connection: keep-alive' \
  -H 'Origin: http://localhost:5173' \
  -H 'Referer: http://localhost:5173/' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0' \
  -H 'content-type: application/json' \
  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Microsoft Edge";v="134"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  --data-raw '{"method":"resources/list","params":{},"jsonrpc":"2.0","id":1}'
```



## 在网页上得到了 schema

![image-20250328151443784](image-20250328151443784.png)



## 安装到 Claude

```bash
mcp install .\mysql_server.py --with mysqlclient
```



Claude 配置文件

```json
{
  "mcpServers": {
    "Demo": {
      "command": "mcp",
      "args": [
        "run",
        "D:\\sjj\\script\\mcp_test\\server.py"
      ]
    },
    "MySQL Explorer": {
      "command": "mcp",
      "args": [
        "run",
        "--with",
        "mysqlclient",
        "D:\\sjj\\script\\mcp_test\\mysql_server.py"
      ]
    }
  }
}
```



## 开始测试

<img src="image-20250328153747577.png" alt="image-20250328153747577"  />

![image-20250328153758113](image-20250328153758113.png)

![image-20250328153822340](image-20250328153822340.png)

![image-20250328153835288](image-20250328153835288.png)

![image-20250328153845382](image-20250328153845382.png)

![image-20250328153857103](image-20250328153857103.png)

![image-20250328153916122](image-20250328153916122.png)

![image-20250328153926680](image-20250328153926680.png)
