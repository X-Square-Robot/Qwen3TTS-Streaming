[English](README.md) | **中文**

# 2026-09-10 engine-only 刷新

本目录记录当前 cursor-enabled engine 的刷新结果，run id 为
`20260910T120301Z`。它与父目录中的历史 engine/Triton 混合数据集分开保存。

- `conditions.json` 记录 artifact、能力路由、环境和采样合同。
- `raw_requests.csv` 包含 11,674 条请求记录，包括 warmup；全部请求成功。
- `summary.csv` 包含 engine gRPC 与原生 WebSocket 的 SDK 客户端统计。
- `nvidia_smi.csv`、`engine_image.txt`、`engine_health.json` 和
  `engine.yaml.snapshot` 保留基础环境快照。

本轮并发档位为 1、8、16、32、64、128，每档 20 个测量轮次和 3 个 warmup
轮次；连接隔离对 `reuse` 与 `cold` 各采集 50 个测量轮次和 5 个 warmup 轮次。

本轮主要观察值：

| 目标 | c1 TTFT 均值 | c128 TTFT 均值 | c128 最大观测 batch |
| --- | ---: | ---: | ---: |
| `engine-grpc` | 24.593 ms | 1,273.570 ms | 128 |
| `engine-websocket` | 23.290 ms | 1,114.293 ms | 128 |

这些是本轮 SDK 客户端 TTFT，不是历史服务端 14.9 ms 声明。SDK WebSocket 连接池已显式
提高到 128，因此 c64/c128 测量的是真实并发 lane。当前 c128 延迟仍显著慢于历史基线，
在发布新的生产吞吐声明前必须调查。Triton 本轮未运行，因此没有出现在这里。

SDK 报告了 client
`0.2.1a2.dev23+g45e5d86.d20260910` 与 engine
`v0.2.1a1-23-g45e5d86-dirty` 的 release version skew；由于协议兼容性和
capabilities 匹配，测试继续执行。
