[English](roadmap.md) | **中文**

# 路线图

下文 v0.1 是历史阶段记录；当前发布线是 v0.2，重点是流式稳定性、原生文本进度和可复现
发布证据。原独立 WebUI 已收敛为由 capabilities 门控的内置 Demo；当前实现与验收以
[`demo_browser_sdk_docs_plan.zh-CN.md`](dev/design/demo_browser_sdk_docs_plan.zh-CN.md)
为准。

目标：把当前高性能工程原型演进为可信、可复现、可协作的高质量开源项目。

## v0.1: 工程预览版（已完成基线）

范围：

- `custom-1.7b` / `custom_voice` 作为唯一推荐稳定路径。
- README、内置 Demo、demo API 明确工程预览版定位。
- benchmark 只发布带完整条件的数字。
- autorun/build/deploy 打通 max batch、max input len、max seq len、dtype、engine mode。
- manifest 记录 engine profile，runtime 启动前校验 profile 上限。
- 内置 Demo 支持真实能力门控和风险提示。
- 中文文档齐全。

退出标准：

- `custom-1.7b` 完成最小端到端验收。
- 关键脚本 `bash -n` 通过。
- Python 单测通过或已记录阻塞原因。
- 内置 Demo 能构建。
- README 不再宣传未测通路径为稳定可用。

## v0.2: 稳定性版本（当前）

重点：

- 发布已验证的重训 checkpoint 和运行时防护，大幅抑制历史跑飞型幻觉问题。
- 将 0/100 确定性探针证据以及后续回归证据绑定到精确 checkpoint、engine profile、
  采样配置和请求语料。
- 维护覆盖短句、长句、数字、英文、中英混排、标点密集和长段落的文本集合。
- 继续改进 streaming TN、spliter、EOS/pad、cache 与采样默认值。
- 每次 release 都发布已知限制和可复现输入。

退出标准：

- `custom-1.7b` / `custom_voice` 仍是 v0.2 推荐稳定路径。
- 已验证的全 bf16 引擎记录了 0/100 确定性跑飞探针结果。
- 每次 release 都能给出已知问题、复现输入和 benchmark 条件。
- 默认参数下严重幻觉/重复已显著降低；任意 checkpoint 仍属于调用方/部署方验收范围。

## v0.3: base / ICL 语音克隆

重点：

- 完成 ref audio preprocessing。
- 打通 speaker embedding、ref codes、ref codec sum vec 到 prefill/build plan。
- 区分 x-vector clone 和 ICL clone 的请求协议。
- 增加 base/ICL 端到端测试。
- 内置 Demo 增加参考音频上传和 ref text 输入，但默认仍标注实验状态。

退出标准：

- base voice clone 可跑通真实 ref audio。
- ICL voice clone 可跑通 ref audio + ref text。
- 错误提示能清楚说明缺失字段或未启用能力。

## v0.4: voice design

重点：

- 完整验证 `design-1.7b`。
- 梳理 instruct 字段、speaker 字段和 custom voice 的互斥关系。
- 建立 voice design 示例集。
- 明确它和 custom voice 的质量边界。

退出标准：

- voice design 有独立 demo 和测试输入。
- README 可以从“实验路径”升级为“可试用路径”。

## v0.5: 部署与可维护性

重点：

- 优化 engine Docker 的环境层/代码层体验。
- 补 K8s/Helm 或 production compose 示例。
- 增加健康检查、限流、日志、metrics、trace id。
- 完善 CI：Python 单测、manifest schema 校验、bash 语法、内置 Demo build。
- 发布版本化 artifact 和 release note。

退出标准：

- 开发期不需要因普通代码改动重建依赖镜像。
- CI 能阻止 README 口径、manifest schema、Demo 类型错误和核心单测回归。

## 英文文档

英文版不作为当前阶段优先项。中文 README 和 docs 稳定后，再翻译：

- README
- known limitations
- deployment
- benchmark methodology
- roadmap

翻译时不能弱化风险提示，也不能把实验路径写成稳定能力。
