**English** | [中文](cross_host_build.zh-CN.md)

# Cross-host TensorRT Engine Build

A TensorRT `.engine` is bound to the target GPU architecture, the TensorRT version, and the build profile. The export host or packaging host must not assume it is identical to the production host; the recommended flow is: export ONNX on the current machine, build the engine on a production-homogeneous target GPU, then bring the engine artifact back to the current machine to assemble the model package and runtime image. Starting the service is a separate step that should only be run when the current machine is itself the production serving host or a local validation host.

## Offline Bundle Flow

1. Collect the fingerprint on a production-homogeneous target machine:

```bash
bash scripts/bash/autorun.sh probe-target --out target_profile.json
```

2. Copy `target_profile.json` back to the export/build-bundle host:

```bash
bash scripts/bash/autorun.sh make-bundle -m custom-1.7b \
  --target-profile target_profile.json \
  --out workspace/engine_build_bundle.tar.zst
```

3. Copy `engine_build_bundle.tar.zst` to the target machine and run:

```bash
mkdir -p /tmp/qwen3-engine-build
tar --zstd -xf engine_build_bundle.tar.zst -C /tmp/qwen3-engine-build
cd /tmp/qwen3-engine-build
bash run.sh
```

4. Copy `engine_artifact_bundle.tar.zst` back to the packaging host and import it:

```bash
bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst
```

5. Back on the packaging host, assemble the deployment artifacts without starting the service:

```bash
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker
```

`package --gateway engine-docker` assembles `workspace/model_repository/tts_orchestrator/<version>` and rebuilds the engine image from the current checkout. The image tag is by default derived from the NGC tag in the Phase B manifest, for example `qwen3-engine:25.03`.

6. Start the service only when the current machine is the one that will actually serve:

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

## Remote SSH Flow

When the packaging host can SSH into the target machine, the bundle flow can be automated:

```bash
bash scripts/bash/autorun.sh remote-build -m custom-1.7b \
  --target-profile target_profile.json \
  --remote-host user@your-gpu-host \
  --remote-workdir /tmp/qwen3-engine-build
```

This command builds the bundle locally, uploads it to the remote, runs `build_on_target.sh` remotely, pulls the artifact back, and imports it into `workspace/exported/`.

## Where the NGC Tag Comes From

In the cross-host scenario, `target_profile.json` is the single source of truth for the NGC tag:

- `probe-target` selects `recommended_ngc_tag` on the target machine based on the production driver.
- `make-bundle` uses that tag to write `build_manifest.json`.
- The engine is built with the same NGC image.
- Phase C package/run derives the runtime image from the manifest, and no longer falls back to guessing from the packaging host's local driver.

## Strict Fingerprint Validation

After importing an artifact, `workspace/exported/artifact_manifest.json` is written. In TRT mode, `package` and `run` validate:

- `ngc_tag`
- `tensorrt_version` major.minor
- `gpu_sm`
- `engine_dtype`
- `max_batch_size`
- `max_input_len`
- `max_seq_len`

Older local builds without an artifact manifest remain compatible and only print a warning. An artifact imported through the cross-host flow that does not match fails outright; for development and debugging you can explicitly set `--allow-fingerprint-mismatch`:

```bash
bash scripts/bash/autorun.sh import-artifact bundle.tar.zst --allow-fingerprint-mismatch
```

## Relationship to Phase C

`package` and `deploy` have different responsibilities:

```bash
# Packaging host / release pipeline: produce only the model package and runtime image
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker

# Serving host / local validation: start the service from an existing model package and runtime image
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

For a local deployment you can directly run `bash scripts/bash/autorun.sh all -m custom-1.7b --gateway engine-docker`, which runs through `setup → build → package → deploy`; for a cross-host deployment, you should usually not run the final `deploy` on the packaging host.

## Command Reference

| Operation | Command |
|------|------|
| Collect target fingerprint | `bash scripts/bash/autorun.sh probe-target --out target_profile.json` |
| Create build bundle | `bash scripts/bash/autorun.sh make-bundle -m custom-1.7b --target-profile p.json` |
| Remote build over SSH | `bash scripts/bash/autorun.sh remote-build -m custom-1.7b --target-profile p.json --remote-host user@host` |
| Import build artifact | `bash scripts/bash/autorun.sh import-artifact bundle.tar.zst` |
| Local build | `bash scripts/bash/autorun.sh build -m custom-1.7b` |
