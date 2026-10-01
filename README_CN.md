# ComfyUI-SelfLift-Split（中文说明）

[English README](README.md)

这是一个独立的 MiniMax H3 音视频 SelfLift 拆分实现：前段使用低分辨率 latent 执行 Euler 采样，将边界处的去噪视频交给外部 latent 放大模型或简单插值节点，在共享的边界 sigma 上重新准备 AV latent，最后交给原生高分辨率采样器完成剩余采样。

本仓库与原版 `comfyui-SelfLift` 独立存在，不修改也不替换原项目。

## 原项目致谢

本项目的分辨率过渡思路和边界采样逻辑基于 **facok** 的原版 [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)，研究背景来自 SelfLift 论文：

- [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)

感谢原项目提供了 SelfLift 的 ComfyUI 实现和 H3 适配方向。本项目将 H3 音视频路径单独拆出，使低分采样、latent 放大、AV 边界交接以及原生二采可以分别检查和调整。

## 新项目新增内容

相对于直接使用原版 SelfLift，本项目新增了一个显式、模块化的 H3 AV 拆分路径：

- `SelfLiftSplitLowH3`：使用标准 Euler 执行前段低分采样，并捕获过渡所需的采样状态和去噪视频边界。
- `SelfLiftSplitTransitionH3`：检查高分模型与 latent 契约，在共享 sigma 处重新准备视频边界，保留音频流，并输出可直接续采的 AV latent 和剩余 sigmas。
- `SelfLiftSplitResizeH3`：提供 H3 视频专用的简单 latent 插值路径。
- 支持在 Low 和 Transition 之间接入外部 learned H3 latent upscaler。
- Low 节点直接输出最终目标像素宽高，目标分辨率模式的放大节点不再需要用户手算尺寸。
- 执行二采前检查 H3 嵌套视频/音频 latent 形状、空间尺寸、采样器类型、sigma 调度、模型采样类型、latent format 和 mask 形状。
- 视频放大期间保持音频为独立流，视频边界准备完成后再重新打包为 AV latent。
- 将边界模式作为节点选项公开，而不是隐藏在一个复合采样节点里。

本仓库专注于 H3 音视频拆分实现，不是原版图像 SelfLift 节点的完整 fork，也不包含原项目独立的 TST 实现。

## 节点说明

### `SelfLiftSplitLowH3`

执行低分前段采样，并捕获交接给 Transition 的边界状态。

- 输入：嵌套 H3 AV latent，视频形状为 `[B, 24, T, H, W]`，音频形状为 `[B, 32, 2, T]`。
- 采样器：仅支持标准 Euler，且 `s_churn=0`。
- `transition_step`：放大前执行的低分 Euler 更新次数。
- `lowres_scale`：低分视频前缀使用的空间缩放比例。
- 输出：采样状态、纯视频 clean latent、缩放后的正负 conditioning 以及最终目标像素尺寸。

`clean_video` 是边界处的去噪视频，也就是 `x0` / `denoised_output` 类型的数据，不是带噪的采样器 `output`。它就是外部视频 latent 放大节点应该接收的输入。

### `SelfLiftSplitTransitionH3`

准备高分 AV latent 和剩余 sigma 尾段。

- 接收放大后的纯视频 latent；
- 保留低分采样器输出中的音频流；
- 输出 `resume_latent`、`tail_sigmas` 和经过验证的高分模型；
- 可选使用 `rho`、`w_min`、`w_max` 以及模型 VAE 做像素锚点修正。

边界模式：

| 模式 | 使用场景 | 视频边界处理 |
| --- | --- | --- |
| `legacy_seed` | learned latent 放大路线，以及对齐参考工作流风格 | 使用原 seed 在 `sigma_next` 处对放大视频重新加噪 |
| `upstream` | 对照原版 SelfLift 边界逻辑 | 使用 `seed + 1` 在 `sigma_k` 构造边界，再用 Euler 重建到 `sigma_next` |
| `simple_interpolation` | 与提供的参考工作流一致的简单插值路线 | 插值后在边界 sigma 执行一次重噪适配 |
| `simple_no_noise` | 不添加视频随机噪声的诊断/对照路线 | 边界处不加入视频随机噪声 |

这些模式不是数学等价的采样方式，而是针对不同路线和对照实验的边界构造方法。

### `SelfLiftSplitResizeH3`

将纯视频 clean latent 插值到目标 H3 latent 网格。它不处理音频，也不执行去噪。当前参考简单插值工作流使用 `nearest-exact`；这个节点只是直接插值工具，不是 learned upscaler。

## 二采接线

Transition 已经准备好二采起点 latent，并在需要时处理路线对应的视频边界噪声。因此原生第二次采样器必须使用 `DisableNoise`：

```text
SelfLiftSplitTransitionH3.resume_latent -> SamplerCustomAdvanced.latent_image
SelfLiftSplitTransitionH3.tail_sigmas   -> SamplerCustomAdvanced.sigmas
DisableNoise                            -> SamplerCustomAdvanced.noise
```

Transition 之后不要再把 `RandomNoise` 接到二采。否则会在已经准备好的二采起点上重复注入随机噪声，导致视频或音频续接异常。

## 包含的工作流

`workflows/` 中包含最终测试使用的两份 H3 示例：

1. **latent 放大模型路线**
   - 使用 `MinimaxH3LatentUpscaler3D`；
   - Transition 使用 `legacy_seed`；
   - 用于 learned H3 latent 放大路线。

2. **简单插值路线**
   - 使用 `SelfLiftSplitResizeH3`，插值方法为 `nearest-exact`；
   - Transition 使用 `simple_interpolation`；
   - 对齐参考工作流，在二采前执行一次边界 sigma 视频适配；
   - 简单插值的画质上限低于 learned 放大，一般需要给高分二采保留足够步数来修复插值误差。

工作流 JSON 依赖用户环境中已经安装的 ComfyUI 核心和 H3 自定义节点。learned 示例还需要安装 `MinimaxH3LatentUpscaler3D` 节点及其模型权重。

## 安装

将本仓库复制或克隆到 ComfyUI 的 `custom_nodes` 目录：

```text
ComfyUI/custom_nodes/ComfyUI-SelfLift-Split
```

重启 ComfyUI。节点位于 `selflift/split` 分类下。

本项目依赖 ComfyUI 提供的 Python 模块，必须在 ComfyUI 环境中运行，不是独立 Python 包。

## 验证

仓库包含数学边界辅助函数、修正函数、节点契约、嵌套 AV 打包、音频交接、mask 和全部边界模式的单元测试。

请使用 ComfyUI 实际使用的 Python 环境运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

最终本地验证共通过 16 项测试。

## 限制

- Low 节点只接受 `s_churn=0` 的标准 Euler。
- 低分和高分模型必须使用相同的 rectified-flow 采样类型、latent format 和 noise scale。
- 外部放大节点必须输出目标空间尺寸的纯视频 latent，不能输出嵌套 AV latent。
- 本项目不声称简单插值能够恢复放大过程中丢失的细节；插值伪影主要依靠高分二采修复。
- `simple_no_noise` 适合诊断和对照，但不等于参考工作流中的简单插值路线。
- Transition 之后的二采接 `RandomNoise` 是错误接法。
- 原版 SelfLift 的图像路径和原项目独立的 TST 功能不属于本仓库。

## 许可证与上游说明

本仓库是针对 MiniMax H3 音视频工作流的独立适配。上游实现和研究背景请参阅 [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) 及 SelfLift 论文。重新分发衍生代码时，请同时遵守上游项目的许可证和声明。
