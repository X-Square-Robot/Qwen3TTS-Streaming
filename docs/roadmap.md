# 路线图

目标：把当前高性能工程原型演进为可信、可复现、可协作的高质量开源项目。

## v0.1: 工程预览版

范围：

- `custom-1.7b` / `custom_voice` 作为唯一推荐稳定路径。
- README、WebUI、demo API 明确工程预览版定位。
- benchmark 只发布带完整条件的数字。
- autorun/build/deploy 打通 max batch、max input len、max seq len、dtype、engine mode。
- manifest 记录 engine profile，runtime 启动前校验 profile 上限。
- WebUI 支持 fixture/live 来源区分和风险提示。
- 中文文档齐全。

退出标准：

- `custom-1.7b` 完成最小端到端验收。
- 关键脚本 `bash -n` 通过。
- Python 单测通过或已记录阻塞原因。
- WebUI 能构建。
- README 不再宣传未测通路径为稳定可用。

## v0.2: 稳定性专项

重点：

- 系统性定位流式幻觉、重复、漏读、插入内容问题。
- 建立文本集合：短句、长句、数字、英文、中英混排、标点密集、长段落。
- 增加音频质量回归测试和人工验收表。
- 改进 spliter、EOS/pad、cache、采样默认值。
- 把失败案例沉淀到 `docs/streaming_hallucination_investigation.md` 或新的中文文档。

退出标准：

- 公开一组稳定性测试集。
- 每次 release 都能给出已知问题和复现输入。
- 默认参数下严重幻觉/重复概率显著降低。

## v0.3: base / ICL 语音克隆

重点：

- 完成 ref audio preprocessing。
- 打通 speaker embedding、ref codes、ref codec sum vec 到 prefill/build plan。
- 区分 x-vector clone 和 ICL clone 的请求协议。
- 增加 base/ICL 端到端测试。
- WebUI 增加参考音频上传和 ref text 输入，但默认仍标注实验状态。

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
- 完善 CI：Python 单测、manifest schema 校验、bash 语法、WebUI build。
- 发布版本化 artifact 和 release note。

退出标准：

- 开发期不需要因普通代码改动重建依赖镜像。
- CI 能阻止 README 口径、manifest schema、WebUI 类型错误和核心单测回归。

## 英文文档

英文版不作为当前阶段优先项。中文 README 和 docs 稳定后，再翻译：

- README
- known limitations
- deployment
- benchmark methodology
- roadmap

翻译时不能弱化风险提示，也不能把实验路径写成稳定能力。
