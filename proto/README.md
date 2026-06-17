# Protocol Definitions

This directory is the **single source of truth** for protocol definitions used by the Qwen3-TTS Triton project.

## Files

| File | Description |
|------|-------------|
| `tts.proto` | gRPC service definition for the TTS engine |
| `tts_pb2.py` | Generated Python protobuf message classes |
| `tts_pb2_grpc.py` | Generated Python gRPC stubs |

## Regenerating

After editing `tts.proto`, regenerate the Python files:

```bash
make proto
# or manually:
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/tts.proto
```

After regeneration, copy the output to consumers:

```bash
make proto-sync
```

This copies the generated files to:
- `engine/gateway/` (engine runtime)
- `client/src/qwen3_tts_client/_proto/` (client SDK)
