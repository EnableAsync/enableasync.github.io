---
title: Higress AI Wasm 插件开发记录
categories: higress
typora-root-url: ./higress
---

# Higress 背景

Higress 是基于阿里内部两年多的 Envoy Gateway 实践沉淀，以开源 Istio 与 Envoy 为核心构建的云原生 API 网关。

<img src="overview.png" alt="image" style="zoom: 33%;" />

Higress 实现了安全防护网关、流量网关、微服务网关三层网关合一，可以显著降低网关的部署和运维成本。

<img src="云原生网关.png" alt="image" style="zoom:50%;" />

# Higress 与大模型

以 ChatGPT 为代表的 AIGC（人工智能生成内容）技术正在引领企业生产的巨大变革，并在企业应用开发领域中占据了重要地位。AI 大模型凭借其卓越的学习和理解能力，可以辅助完成各种复杂的任务，极大地提升了工作效率和创新能力。

在软件开发领域，AI 大模型能够显著提高开发人员的工作效率。它们可以协助编写和调试代码，自动生成测试用例，并提供最佳实践建议，从而加速开发周期，降低错误率。科研领域的研究人员则利用这些模型快速获取和理解最新的科研进展，自动化文献综述和数据分析，节省大量时间和精力。

然而，随着 AI 大模型在企业中的应用不断深入，许多企业开始探索如何降低这些技术的使用成本。一个常见的解决方案是通过网关进行 AI 大模型的 API 管理。这样的管理方式不仅能够集中控制和优化模型的调用频率和资源使用，还可以保障数据安全和隐私合规。通过网关，企业能够灵活地调整使用策略，以更低的成本享受 AI 技术带来的高效益。

Higress 前瞻性地通过 Wasm 实现了LLM Proxy 插件和 AI Assistant 插件帮助开发者快速构建 RAG 应用。

# 为 Higress 开发 ai-cache 插件的意义

<img src="arch.png" alt="image-20240826154248690" style="zoom: 33%;" />

AI 缓存插件的目标是在构建 AI 应用时，通过智能缓存机制，**减少对LLM提供商API的请求数量，从而降低使用成本，同时确保返回结果的质量**。

在具体实现过程中，Higress AI-Cache 利用向量相似度技术，通过分析和比较用户查询的特征向量，与缓存中已有的查询结果进行匹配。当用户发起新的查询时，Higress 首先计算该查询的向量表示，并在缓存中寻找相似度较高的结果。如果找到足够相似的缓存结果，插件将直接返回该结果，而无需再向LLM提供商API发出新的请求。

该插件适用于多个LLM提供商API，例如通义千问、moonshot、OpenAI等。通过集成这些API，插件可以在不同的AI应用场景中灵活应用，包括但不限于智能客服、内容生成、代码编写和调试等领域。

Higress AI 缓存插件核心优势在于，通过减少不必要的API调用，显著降低了使用大型语言模型的成本。同时，向量相似度技术确保了缓存结果的准确性和相关性，使得用户体验不受影响。插件还具备动态更新和管理缓存的功能，能够根据查询频率和变化情况，自动调整缓存策略，以保持最佳性能。

# AI-Cache 插件运行流程

<img src="seq.png" alt="image-20240826155252868" style="zoom:50%;" />

1. 当用户请求 LLM API 时，首先在 AI-Cache 插件中对用户的请求内容进行 Embedding 操作，将用户的请求转换为向量数据。
2. 之后在向量数据库中进行相似度搜索。当相似度高于预先设定的阈值时直接返回 Cache 中的内容，不请求 LLM API。否则执行下一个步骤。
3. 当相似度低于预先设定的阈值时，使用 AI-Proxy 插件进行请求转发，向 LLM API 发送请求。并将请求结果缓存在本地的向量数据库中，并把结果返回给用户。

# Docker compose 部署 Higress 进行插件开发

使用 K8s 进行 Higress 开发时复杂度较高，这里介绍使用 Docker compose 开发的方法。

## docker-compose 文件编写

首先是 `docker-compose.yml` 文件的编写，这里给出我使用的文件内容，其中的要点主要有：

1. 对 higress 的环境变量设置：`--component-log-level wasm:debug` 和 `/etc/envoy/envoy.yaml` 分别指定日志级别和配置文件。
2. wasm 文件和 envoy.yaml 文件的映射。
3. higress 的上游服务需要和 higress 在同一个 network 中。
4. higress 的上游服务需要在 higress 之前启动完成。

```yaml
version: '3.7'
services:
  envoy:
    # image: higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/gateway:v1.4.0-rc.1
    image: higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/gateway:1.4.2
    entrypoint: /usr/local/bin/envoy
    # 注意这里对wasm开启了debug级别日志，正式部署时则默认info级别
    command: -c /etc/envoy/envoy.yaml --component-log-level wasm:debug
    depends_on:
    - httpbin
    - redis
    - chroma
    - es
    networks:
    - wasmtest
    ports:
    - "10000:10000"
    - "9901:9901"
    volumes:
    - ./envoy.yaml:/etc/envoy/envoy.yaml
    # 注意默认没有这两个 wasm 的时候，docker 会创建文件夹，这样会出错，需要有 wasm 文件之后 down 然后重新 up
    - ./ai-cache.wasm:/etc/envoy/ai-cache.wasm
    - ./ai-proxy.wasm:/etc/envoy/ai-proxy.wasm

  chroma:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma

  redis:
    image: redis:latest
    networks:
    - wasmtest
    ports:
    - "6379:6379"

  httpbin:
    image: kennethreitz/httpbin:latest
    networks:
    - wasmtest
    ports:
    - "12345:80"

  es:
    image: elasticsearch:8.15.0
    environment:
      - "TZ=Asia/Shanghai"
      - "discovery.type=single-node"
      - "xpack.security.http.ssl.enabled=false"
      - "xpack.license.self_generated.type=trial"
      - "ELASTIC_PASSWORD=123456"
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - wasmtest

volumes:
  weaviate_data: {}
  chroma-data:
    driver: local

networks:
  wasmtest: {}
```

## envoy.yaml 文件编写

在这里踩过许多坑，一一记录下来：

1. 在 wasm 插件中如果需要请求外部服务，需要在 `envoy.yaml` 中的 `clusters` 中一一指定并使用 `cluster_name` 访问，比如需要访问远程的 Dashscope Embedding 接口，则需要创建 `cluster_name` 名为 `outbound|443||dashvector.dns` 的 cluster，之后在代码中通过以下方式访问：

```go
client := wrapper.NewClusterClient(wrapper.DnsCluster{
			ServiceName: c.serviceName,
			Port:        c.servicePort,
			Domain:      c.serviceHost,
		})
```

   这里的 client 支持以下方法：

```go
type HttpClient interface {
	Get(path string, headers [][2]string, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Head(path string, headers [][2]string, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Options(path string, headers [][2]string, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Post(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Put(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Patch(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Delete(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Connect(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Trace(path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
	Call(method, path string, headers [][2]string, body []byte, cb ResponseCallback, timeoutMillisecond ...uint32) error
}
```

   **注意，这里的 HttpClient 是异步的，所以如果需要对结果进行处理之后再继续进行，则需要把逻辑写在 `ResponseCallback` 中。**

2. 如果请求的服务是 HTTPS，则需要在 `cluster` 中指定是 `tls` 以及服务对应的 `sni`。
2. envoy.yaml 里配置 Redis cluster 时，socketAddr 要尽量用 IP，不要用主机名。详细原因在 Wasm 插件编写中解释。

```yaml
admin:
  address:
    socket_address:
      protocol: TCP
      address: 0.0.0.0
      port_value: 9901
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        protocol: TCP
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      # httpbin
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          scheme_header_transformation:
            scheme_to_overwrite: https
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: llm
                  timeout: 300s

          http_filters:
          # ai-cache
          - name: ai-cache
            typed_config:
              "@type": type.googleapis.com/udpa.type.v1.TypedStruct
              type_url: type.googleapis.com/envoy.extensions.filters.http.wasm.v3.Wasm
              value:
                config:
                  name: ai-cache
                  vm_config:
                    runtime: envoy.wasm.runtime.v8
                    code:
                      local:
                        filename: /etc/envoy/ai-cache.wasm
                  configuration:
                    "@type": "type.googleapis.com/google.protobuf.StringValue"
                    value: |
                      {
                        "embeddingProvider": {
                          "type": "dashscope",
                          "serviceName": "dashscope",
                          "apiKey": "sk-key",
                          "DashScopeServiceName": "dashscope"
                        },
                        "vectorProvider": {
                          "VectorStoreProviderType": "elasticsearch",
                          "ThresholdRelation": "gte",
                          "ESThreshold": 0.7,
                          "ESServiceName": "es",
                          "ESIndex": "higress",
                          "ESUsername": "elastic",
                          "ESPassword": "123456"
                        },
                        "cacheKeyFrom": {
                          "requestBody": ""
                        },
                        "cacheValueFrom": {
                          "responseBody": ""
                        },
                        "cacheStreamValueFrom": {
                          "responseBody": ""
                        },
                        "returnResponseTemplate": "",
                        "returnTestResponseTemplate": "",
                        "ReturnStreamResponseTemplate": "",
                        "redis": {
                          "serviceName": "redis_cluster",
                          "timeout": 2000
                        }
                      }

          # 上面的配置中 redis 的配置名字是 redis，而不是 golang tag 中的 redisConfig
                        # "vectorProvider": {
                        #   "VectorStoreProviderType": "chroma",
                        #   "ChromaServiceName": "chroma",
                        #   "ChromaCollectionID": "0294deb1-8ef5-4582-b21c-75f23093db2c"
                        # },

                        # "vectorProvider": {
                        #   "VectorStoreProviderType": "elasticsearch",
                        #   "ThresholdRelation": "gte",
                        #   "ESThreshold": 0.7,
                        #   "ESServiceName": "es",
                        #   "ESIndex": "higress",
                        #   "ESUsername": "elastic",
                        #   "ESPassword": "123456"
                        # },
          # llm-proxy
          - name: llm-proxy
            typed_config:
              "@type": type.googleapis.com/udpa.type.v1.TypedStruct
              type_url: type.googleapis.com/envoy.extensions.filters.http.wasm.v3.Wasm
              value:
                config:
                  name: llm
                  vm_config:
                    runtime: envoy.wasm.runtime.v8
                    code:
                      local:
                        filename: /etc/envoy/ai-proxy.wasm
                  configuration:
                    "@type": "type.googleapis.com/google.protobuf.StringValue"
                    value: | # 插件配置
                      {
                        "provider": {
                          "type": "openai",                                
                          "apiTokens": [
                            "YOUR_API_TOKEN"
                          ],
                          "openaiCustomUrl": "172.17.0.1:8000/v1/chat/completions"
                        }
                      }


          - name: envoy.filters.http.router
      
  clusters:
  - name: httpbin
    connect_timeout: 30s
    type: LOGICAL_DNS
    # Comment out the following line to test on v6 networks
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: httpbin
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: httpbin
                port_value: 80
  - name: outbound|6379||redis_cluster
    connect_timeout: 1s
    type: strict_dns
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: outbound|6379||redis_cluster
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 172.17.0.1
                    port_value: 6379
    typed_extension_protocol_options:
      envoy.filters.network.redis_proxy:
        "@type": type.googleapis.com/envoy.extensions.filters.network.redis_proxy.v3.RedisProtocolOptions
  # chroma
  - name: outbound|8001||chroma.dns
    connect_timeout: 30s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: outbound|8001||chroma.dns
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 172.17.0.1 # 本地 API 服务地址，这里是 docker0
                    port_value: 8001

  # es
  - name: outbound|9200||es.dns
    connect_timeout: 30s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: outbound|9200||es.dns
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 172.17.0.1 # 本地 API 服务地址，这里是 docker0
                    port_value: 9200

  # llm
  - name: llm
    connect_timeout: 30s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: llm
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: 172.17.0.1 # 本地 API 服务地址，这里是 docker0
                    port_value: 8000
  # dashvector
  - name: outbound|443||dashvector.dns
    connect_timeout: 30s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: outbound|443||dashvector.dns
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: vrs-cn-0dw3vnaqs0002z.dashvector.cn-hangzhou.aliyuncs.com
                    port_value: 443
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
        "sni": "vrs-cn-0dw3vnaqs0002z.dashvector.cn-hangzhou.aliyuncs.com"
  # dashscope
  - name: outbound|443||dashscope.dns
    connect_timeout: 30s
    type: LOGICAL_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: outbound|443||dashscope.dns
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: dashscope.aliyuncs.com
                    port_value: 443
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
        "sni": "dashscope.aliyuncs.com"
```

## Wasm 插件编写

### Wasm 插件中无法使用 golang 的 `net/http` 库发送请求，必须使用 higress 封装的 HTTP client

### Wasm 插件请求 Redis 时，提示 "bad argument" 错误

   解决办法：envoy.yaml 里配置 Redis cluster 时，socketAddr 要用 IP，不要用主机名。

   在开发 Wasm 插件过程中，我们镜像会使用 Docker Compose + Envoy + Volume Mount 的方式测试本地构建出来的插件。如果插件需要连接 Redis，那么我们就需要在 envoy.yaml 中配置一个 Redis 的 cluster。如果配置中的 Redis 节点地址使用机器名，那么在启动插件的时候可能会出现初始化 Redis 客户端报“bad argument”的错误。

   原因：这种错误一般只发生在插件在 `parseConfig` 阶段调用 `RedisClusterClient.Init()` 函数的时候。、

   在 Envoy 初始化的过程中，集群信息的初始化与 Wasm 插件的初始化可以认为是并行进行的。如果使用主机名进行配置，要获取实例的实际 IP 就需要经过 DNS 解析。而 DNS 解析一般是需要一些时间的，Redis 客户端的初始化又需要与 Redis 集群建立连接和通信。这一延迟就可能会导致 Wasm 插件进行初始化时 Redis 的集群信息还没有就绪，进而引发上述报错。

   而在 Higress 的实际运行过程中，集群信息是通过 xDS 进行下发的，这个延迟的问题不会非常显著。

### ` proxywasm.ResumeHttpRequest()` 的使用

下面是一个 wasm 插件访问外部请求并返回给下游的例子：

```go
func onHttpRequestHeaders(ctx wrapper.HttpContext, config MyConfig, log wrapper.Log) types.Action {
  // 使用client的Get方法发起HTTP Get调用，此处省略了timeout参数，默认超时时间500毫秒
  config.client.Get(config.requestPath, nil,
    // 回调函数，将在响应异步返回时被执行
    func(statusCode int, responseHeaders http.Header, responseBody []byte) {
      // 请求没有返回200状态码，进行处理
      if statusCode != http.StatusOK {
        log.Errorf("http call failed, status: %d", statusCode)
        proxywasm.SendHttpResponse(http.StatusInternalServerError, nil,
          []byte("http call failed"), -1)
        return
      }
      // 打印响应的HTTP状态码和应答body
      log.Infof("get status: %d, response body: %s", statusCode, responseBody)
      // 从应答头中解析token字段设置到原始请求头中
      token := responseHeaders.Get(config.tokenHeader)
      if token != "" {
        proxywasm.AddHttpRequestHeader(config.tokenHeader, token)
      }
      // 恢复原始请求流程，继续往下处理，才能正常转发给后端服务
      proxywasm.ResumeHttpRequest()
    })
  // 需要等待异步回调完成，返回Pause状态，可以被ResumeHttpRequest恢复
  return types.ActionPause
}
```

在这里需要注意的是 `onHttpRequestHeaders` 方法返回了 `types.ActionPause` 等待了 `Get` 方法，我们前面说过 `client.Get` 是一个异步请求，如果不显式地进行等待，那么下游无法得到 higress 的请求结果。因此这里返回 `types.ActionPause` 等待请求完成之后，在 `client.Get` 的 response callback 函数中调用 ` proxywasm.ResumeHttpRequest()` 恢复原始请求流程，继续往下处理，才能正常转发给后端服务。



