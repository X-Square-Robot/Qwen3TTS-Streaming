# 项目协作与 Agent 工作规范

任何开发者或自动化 Agent 在修改代码、文档、版本、分支或发布配置前，都必须遵守本规范。

## 开发约定

- 使用虚拟环境运行代码，不要修改或污染 base 环境。
- 不重复造轮子。优先采用维护良好、符合行业惯例的库；不稳定的个人库不视为默认依赖。
- 协议遵循行业惯例，不随意发明私有字符串协议；动作和协议类型优先使用枚举。
- 一系列局部问题如果来自同一设计缺陷，应先给出设计解，不能持续拆东墙补西墙。
- 单元、合同、集成和 E2E 测试应与改动风险匹配，禁止未经验证填充实现。
- 复杂模块必须按职责合理拆分并形成窄接口，不能形成巨型文件或无归属碎片。
- 代码应合理抽象和复用，测试工具也必须按职责归类。

## 代码组织约束

- 顶层目录表达业务领域或运行时角色。
- 每个组件只拥有一个明确问题，并提供窄公开接口。
- 组件内部继续拆成子问题，直到叶子模块和函数职责单一。
- 依赖必须单向，跨组件只能经过公开合同。
- `main`、bootstrap 等入口只负责装配，不承载业务逻辑。
- 文件大小是职责划分的结果，不能为了拆文件制造无归属碎片。
- 测试镜像 owner 结构，单元、合同、集成、E2E 明确分层。
- 看到路径应知道谁负责，看到 import 应知道依赖谁，看到测试位置应知道验证哪一层。

## 原生游标、TRT 与流式 TN 的不可违背约束

修改下列范围前必须先完整阅读：

- `docs/dev/architecture/native_cursor_trt_fusion_decision.zh-CN.md`
- `docs/dev/design/incremental_text_normalization_and_soft_drain.zh-CN.md`

涉及范围包括：

- `scripts/export/` 的 Talker、Code Predictor、Code2Wav 或 fused graph；
- `engine/backend/` 的 TRT executor、slot state、prefill、decode 和 batching；
- `engine/frontend/` 的 streaming TN、text commitment 和 raw mapping；
- native cursor 模型、文本进度和相关协议。

必须遵循以下规则：

1. **“支持官方 Qwen3-TTS”指 TRT 图和运行时兼容，不是新增官方 PyTorch backend。**
   不得为规避导出兼容问题把官方 PyTorch 模型包装成另一条在线推理路径。

2. **游标头的全部神经网络计算进入 cursor-enabled TRT 图。** 对已验证的
   `custom-1.7B`，text encoder、codec trunk 和 matcher 都属于 TRT 计算；CPU 只做
   TN、状态编排、重锚定、坐标投影和事件发布。不得默认恢复为长期 CPU PyTorch sidecar。
   lookahead pending、膨胀卷积历史激活、动态 label buffer 和 frontier 同步都是这条
   有状态 TRT 路径必须实现的流式工作，不得把它们误判为不能融合的理由。

3. **一个 codec frame 只执行一次 matcher 更新。** 游标只观察该帧的 codebook-0
   token，即 `codec0 == full_codec[:, 0]`。其他 codebook 不分别执行 matcher。当前
   `FUSED_CHUNK_T=1`，所以每次 decode graph 调用对应一次 matcher；未来 `T>1` 时必须
   在图内按帧顺序展开 `T` 次。

4. **游标直接消费图内采样出的 `codec_token_0`。** 显式 `codec0` output 只服务于
   executor EOS、prefill、metrics、debug 和兼容路径，不允许把它拷回 CPU 后再触发第二次
   游标神经推理。读取必须兼容显式 `codec0` 与旧图的 `full_codec[:, 0]`。

5. **动态 TN、Unicode 和 raw/normalized 映射永远留在 CPU。** TRT 只接收数值
   `label_ids`、有效长度和神经状态，不接收 raw Unicode，也不执行 TN。

6. **主 streaming TN 是唯一 spoken-form 真相来源。** 不得把 native-cursor 分支中的
   `IncrementalExpander` 或其他规则复制为第二套在线 TN。游标 label plan 必须从主 TN 的
   `TextCommit`、mapping 和 `CanonicalTextJournal` 派生。

7. **TN 后游标映射回 TN 前文本必须经过稳定 owner spans。** 不能用长度比例或旧 label
   index 猜测 raw 坐标。tail rewrite/reanchor 在 CPU 上通过 owner id/raw span 完成，公开
   high-water 永不回退。

8. **协议 high-water 与展示插值必须分离。** `raw_codepoint_end`、
   `normalized_codepoint_end` 是保守整数边界，允许 TN 字段完成时跳跃；owner 内浮点插值只
   用于 UI，不得用于计费、断点恢复、音频 release 或“已读完”确认。

9. **第一版 native cursor 只保证已验证的 `custom-1.7B` 组合。** 官方 0.6B、Base、
   VoiceDesign 或其他 checkpoint 没有匹配并验收的 cursor head 时，必须运行标准 TRT plan
   并降级 EMA。不得静默挂载不匹配的 2M 头。

10. **标准 plan 与 cursor-enabled plan 必须由 manifest/fingerprint 显式区分。** 至少
    校验 model/checkpoint、codec vocab、codebook 数、帧率、cursor head、label vocab、TN
    rules 和适用语音域。不匹配必须 fail closed 到 native cursor disabled；是否允许 TTS
    服务继续由明确的 fail-open 配置决定。

11. **lookahead 是神经状态时序，不是额外 codec。** 发布头 `right_context=1`，第 `t`
    个 codec0 到来时描述第 `t-1` 帧。正常 EOS decode 已经会生成并丢弃 EOS 音频，EOS
    codec0 可直接完成最后一个语音帧的 pending lookahead；只有异常终止或严格 zero-padding
    对照时才需要 cursor-only/masked flush。任何 flush 都不得提交额外 Talker/Code2Wav/KV/PCM
    状态。

12. **CUDA Graph 固定的是 shape、地址和拓扑，不是输入值。** 动态 TN 产生的新
    `cursor_label_ids` 应写入预分配的固定 `[B, M_max]` buffer，配合 `label_count` mask
    复用已 capture 的 graph。不得因 label 内容变化而把 cursor matcher 拆回 CPU；只有
    shape/profile/地址变化才需要重新 capture 或切换 graph entry。

13. **实现不得仅验证“图能构建”。** 必须以冻结 codec0/label 输入比较 PyTorch 参考、
    ONNX 和 TRT 的逐帧轨迹，并覆盖 prefill、decode、lookahead、EOS、TN 字段重排、tail
    rewrite、并发 slot 隔离和官方无头 fallback。

14. **现有计算 backend 以扩展合同的方式接入游标，不做架构重写。** scheduler、KV cache、
    batching、Code2Wav 和 engine FSM 保持不变；只增加 cursor label/state 输入、fused graph
    分支、slot state 转发和 CPU 后处理。Talker 的 `input_embeds` 不能直接替代已发布游标头
    的 text embedding；图内应使用 cursor 自己的 label embedding。Talker 采样出的内部
    `codec_token_0` 则应直接连接到 cursor trunk。

15. **通过 manifest 能力协商做智能路由，不在图内堆运行时模型分支。** 导出工厂可以生成
    standard TRT plan 和 cursor-enabled TRT plan；engine load/admission 根据 model/head
    fingerprint 选择 graph route，会话再以 `AUTO/NATIVE/EMA/DISABLED` 枚举选择 progress
    route。官方模型没有匹配游标头时必须走 standard plan；native 运行中失败只能降级 EMA，
    不能中途重置 Talker/Code2Wav 状态或切换到官方 PyTorch backend。

16. **`head.pt` 随模型权重发布，不作为仓库资源或 CLI 开关。** 导出器自动发现模型目录
    （过渡期兼容 `model/weights/head.pt`）或已导出的 `<variant>/weights/head.pt`；模型侧
    找到时将其复制到导出权重目录，供 bundle 一起发布。`weights/head.pt` 的存在表示该
    模型请求启用游标，但只有匹配的 cursor-enabled TRT plan 和 manifest fingerprint 才能
    真正启用；没有该文件必须保持 standard plan。源文件与导出副本不一致时必须失败，禁止
    通过 `--cursor-head` 或其他外部路径静默挂载不属于当前权重的游标头。

如需改变以上任一规则，先更新决策记录，写清理由、验证数据、兼容影响和迁移方案，再修改实现。
