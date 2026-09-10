[中文](README.zh-CN.md) | **English**

# 2026-09-10 engine-only refresh

This directory records the current cursor-enabled engine refresh from run
`20260910T120301Z`. It is intentionally separate from the historical mixed
engine/Triton dataset in the parent directory.

- `conditions.json` records the artifact, capability route, environment, and
  sampling contract.
- `raw_requests.csv` contains 11,674 request records, including warmup rows;
  all requests completed successfully.
- `summary.csv` contains the pooled SDK-client statistics for engine gRPC and
  native WebSocket.
- `nvidia_smi.csv`, `engine_image.txt`, `engine_health.json`, and
  `engine.yaml.snapshot` preserve the basic environment snapshot.

The run used concurrency levels 1, 8, 16, 32, 64, and 128 with 20 measured
rounds and 3 warmup rounds per level. Connection isolation used 50 measured
rounds and 5 warmup rounds for both `reuse` and `cold`.

Headline observations from this refresh:

| Target | c1 TTFT avg | c128 TTFT avg | c128 max batch seen |
| --- | ---: | ---: | ---: |
| `engine-grpc` | 24.593 ms | 1,273.570 ms | 128 |
| `engine-websocket` | 23.290 ms | 1,114.293 ms | 128 |

These are SDK-client TTFT values from this run, not the historical server-side
14.9 ms claim. The SDK WebSocket pool was explicitly raised to 128 so the c64
and c128 cells measured actual concurrent lanes. The current c128 latency is
still substantially slower than the historical baseline and must be investigated
before publishing a new production throughput claim. Triton was not running and
is not represented here.

The SDK warned about a release-version skew between client
`0.2.1a2.dev23+g45e5d86.d20260910` and engine
`v0.2.1a1-23-g45e5d86-dirty`; the run continued because protocol compatibility
and advertised capabilities matched.
