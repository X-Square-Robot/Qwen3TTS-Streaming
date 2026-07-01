[English](README.md) | **中文**

# 协议定义

本目录是 Qwen3TTS-Streaming 项目所用协议定义的**单一真相源（single source of truth）**。

## 文件

| File | Description |
|------|-------------|
| `tts.proto` | TTS 引擎的 gRPC 服务定义 |
| `tts_pb2.py` | 生成的 Python protobuf 消息类 |
| `tts_pb2_grpc.py` | 生成的 Python gRPC stub |

## 重新生成

编辑 `tts.proto` 之后，重新生成 Python 文件：

```bash
make proto
# or manually:
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/tts.proto
```

重新生成后，将输出复制到各消费方：

```bash
make proto-sync
```

这会将生成的文件复制到：
- `engine/gateway/`（引擎运行时）
- `client/src/qwen3tts/_proto/`（客户端 SDK）
