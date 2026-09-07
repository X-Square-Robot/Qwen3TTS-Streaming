[English](README.md) | **中文**

# 调用方文档

这组文档与内置 Demo 使用同一份 Markdown 源码，按接入深度分为三层：

runtime 只从 `/demo/` 提供一个浏览器门户（源码在 `web/packages/demo`）。其中
`/demo/#/lab` 承载公共 Realtime 实验；可选的 `demo_api` 只增加已迁移的深度工程面板，
不会再提供第二套前端。

## 01 快速接入

- [5 分钟接入](quickstart.zh-CN.md)——确认端点、安装匹配 SDK、完成第一次合成。

## 02 高级配置

- [高级配置](advanced_configuration.zh-CN.md)——任务、增量文本、音频、VAD、交付策略和鉴权。

## 03 更多细节

- [Python SDK](../../client/README.zh-CN.md)——同步、异步、流式和异常参考。
- [Browser SDK](../../web/packages/browser-sdk/README.zh-CN.md)——网页播放、游标和文件保存。
- [Realtime 接口与事件](realtime_api.zh-CN.md)——仅供自研客户端或协议排查使用。
- [限制与上线检查](known_limitations.zh-CN.md)——正式业务需要承担的产品边界。

自行部署服务的维护者请阅读[部署说明](deployment.zh-CN.md)；引擎贡献者请进入
[开发者文档](../dev/README.zh-CN.md)。它们不进入调用方门户的默认导航。
