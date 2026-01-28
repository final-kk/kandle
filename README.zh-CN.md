<div align="center">

# 🕯️ Kandle

**JavaScript 原生的 PyTorch 风格机器学习框架**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.4.5-blue.svg)](https://www.typescriptlang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-green.svg)](https://www.w3.org/TR/webgpu/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-preview-orange.svg)](#预览版声明)

*为 JavaScript 生态带来真正的 PyTorch 体验*

[快速开始](#-快速开始) • [核心特性](#-核心特性) • [示例项目](#-示例项目) • [架构设计](#-架构设计) • [开发路线](#-开发路线)

</div>

---

## 📖 项目简介

Kandle 是一个 **JavaScript 原生**的机器学习框架，采用 **Eager Mode**（动态图）执行模式，深度参考了 PyTorch 的 ATen/c10 架构设计。我不仅将 PyTorch 视为一个 Python 框架，更将其作为现代 AI 框架的 **API 规范标准**，致力于在 JavaScript 生态中实现与 PyTorch 高度对齐的 API 体系。

### 🎯 核心价值主张

- **🔄 动态图执行**：真正的 Eager Mode，支持逐层调试、中间状态检查和动态控制流
- **🎨 PyTorch API 对齐**：从架构层面对齐，而非简单的 API 包装，降低迁移成本和学习曲线
- **⚡ 混合后端架构**：统一接口下支持 WebGPU（GPU 加速）和纯 JS（CPU 计算）双后端
- **🧩 完整的张量系统**：实现了完整的 Stride 机制、广播、视图操作和非连续内存支持
- **🎵 丰富的算子库**：200+ 张量操作，覆盖算术、线性代数、卷积、FFT、音频处理等领域
- **🚀 开箱即用的模型**：原生支持 Qwen3、Whisper 等主流模型，可直接加载 Safetensor 权重

### 💡 为什么选择 Kandle？

当前 JavaScript 生态下存在诸如 ONNX Runtime、WebLLM等优秀的推理引擎，但它们都是**黑盒系统**，专注于静态图推理。Kandle 作为**白盒框架**，填补了以下空白：

| 需求场景 | 黑盒推理引擎 | Kandle（白盒框架） |
|---------|-------------|-------------------|
| **中间计算过程** | ❌ 静态图编译后无法干预 | ✅ 动态图可在任意层暂停/检查 |
| **模型可解释性** | ❌ 黑盒，无法访问内部状态 | ✅ Hook、逐层状态导出 |
| **自定义计算流** | ❌ 受限于预定义 Pipeline | ✅ 完全可编程的控制流 |
| **预处理/后处理** | ⚠️ 需要额外工具链/导出ONNX | ✅ 统一张量操作体系 |
| **API 学习成本** | ⚠️ 框架专有 API | ✅ PyTorch 用户零成本迁移 |
| **调试体验** | ❌ 黑盒难以定位问题 | ✅ 可打"断点"式逐层调试 |
| **推理性能** | ✅ 静态图全局优化 | ⚠️ Eager Mode 权衡 |

**白盒能做到的，黑盒做不到**：
- 🔬 **逐层特征提取**：在任意层导出中间 Tensor 进行可视化分析
- 🎨 **运行时层替换**：动态替换/跳过某些层，实现模型剪枝或A/B测试
- 🧪 **自定义损失函数**：结合业务逻辑设计特殊的计算路径
- 🎯 **精确控制内存**：手动管理 Tensor 生命周期，优化显存占用
- 🌐 **与 DOM API 深度集成**：Hook 直接绑定 Canvas/WebGL 进行实时渲染

**适用场景**：研究、原型开发、模型调试、需要中间计算的应用、音频/视觉预处理、可解释性分析  
**不适用场景**：生产环境的高性能推理（请使用 ONNX Runtime 或 WebLLM）

---

## 🚨 技术验证原型声明

> ⚠️ **这是一个技术验证原型，而非生产就绪的预览版**

- ✅ 当前版本专注于**前向传播架构验证**，已实现 200+ 算子和完整 nn.Module 系统
- 🚧 **Autograd（反向传播）**正在开发中，将在下一版本完整实现
- ⚠️ **快乐路径免责**：当前实现主要验证主流程（Happy Path），边界情况和错误处理尚不完善
- 🔒 **暂不接受 PR**：当前开发中的分支已与当前公开版本已彻底分叉，并且确定有破坏性变更. 待架构稳定后开放贡献
- 💬 **欢迎反馈**：我当前有些闭门造车，非常期待社区对"JavaScript 版 PyTorch 应该是什么样"的想法和建议
- 🎯 **算子需求收集**：除了源语算子外，我希望了解社区需要早期支持哪些特定算子

---

## 🌐 在线体验

无需安装，即刻体验 Kandle .提供了基于 **Qwen3-0.6B** 的可视化交互 Demo，完整展示了 Eager Mode 框架在**模型可解释性**方面的独特优势：

### 📍 访问地址

- **🤗 HuggingFace Spaces**：[https://huggingface.co/spaces/finalkk/kandle-demo](https://huggingface.co/spaces/finalkk/kandle-demo)
- **⚡ Vercel**：[http://kandle-demo.vercel.app/](http://kandle-demo.vercel.app/)

### ✨ Demo 核心功能

| 功能 | 说明 |
|------|------|
| **🎯 单步执行** | 逐步执行前向传播 |
| **⏮️ 时光倒流** | 单步回退，重新选择生成路径 |
| **🎲 手动干预生成** | 在每个 token 生成时手动选择候选词，探索不同分支 |
| **🔍 Logit Lens** | 可视化每一层的输出在词表空间的概率分布 |
| **🔗 Attention Links** | 交互式查看 Self-Attention 的权重连接关系 |
| **🔥 热力图可视化** | 实时展示 Attention Map、激活值分布 |

> 💡 **这就是白盒框架的意义**：不仅能推理，更能"解剖"每一步计算过程。

### 🎬 使用建议

1. **探索模型思考过程**：单步执行时观察每层输出的 top-k tokens，理解模型如何逐步"聚焦"到最终答案
2. **对比不同路径**：回退后选择不同的候选词，观察生成结果的分叉点
3. **发现 Attention 模式**：通过 Attention Links 发现模型关注的关键 token（如代词指向、上下文依赖）
4. **调试与教学**：适合研究者理解 Transformer 内部机制，或用于教学演示

### ⚠️ demo的一些限制
1. **仅支持原始预训练版本**: 当前没有实现量化等技术, 仅能加载原始bf16版本权重
2. **模型尺寸相对较大** : 原始版本模型大小为1.5G左右, 建议手动下载模型后使用 WebFile 或者 Upload 进行加载  
[Qwen3-0.6B地址](https://huggingface.co/Qwen/Qwen3-0.6B)
---

## 🚀 快速开始

### 安装依赖

```bash
# 浏览器环境仅需安装核心库
# 使用 pnpm（推荐）
pnpm add @kandle/core @kandle/backend-webgpu

# 可选的 type 库, 工具库, 预模型构建工具
pnpm add @kandle/types @kandle/utils @kandle/model-utils

# 或使用 npm
npm install @kandle/core @kandle/backend-webgpu

# 如果需要在 Node.js 环境运行，需额外安装 webgpu polyfill
npm install webgpu

```

### 环境要求

- **Node.js**：≥ 18.0.0（需支持 ES2020+）
- **浏览器**：Chrome/Edge ≥ 113（WebGPU 支持）
- **TypeScript**：≥ 5.0（可选）

### 基础使用示例

#### 1️⃣ 初始化后端（WebGPU）

```typescript
import { env } from "@kandle/core";
import { WebGPUBackend } from "@kandle/backend-webgpu";

export async function initWebGPU() {
    const backend = await WebGPUBackend.create();
    env.setBackend(backend);
    env.setDefaultDevice(backend.name);
}

```

#### 2️⃣ 张量操作与广播机制

```typescript
import * as k from '@kandle/core';
import { Tensor } from '@kandle/core';

// 创建张量
const a = new Tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'float32' });
const b = k.randn([2, 3]);

// 算术运算（支持广播）
const result = a.add(b).mul(2).softmax(-1);

// 获取数据（WebGPU 异步读取）
const data = await result.dataAsync();
console.log(data); // Float32Array [...]

// 形状操作（零拷贝视图）
const transposed = a.transpose(0, 1);
console.log(transposed.shape); // [3, 2]
console.log(a.storageId === transposed.storageId); // true
console.log(a.id === transposed.id); // false
const reshaped = a.reshape([3, 2]);
console.log(reshaped.shape); // [3, 2]
console.log(a.storageId === reshaped.storageId); // true
console.log(a.id === reshaped.id); // false


// 高级索引（Python 风格）
const slicedContiguous = a.slice(":1, 1:"); // a[:1, 1:]
console.log(slicedContiguous.shape) // [1, 2];
console.log(a.storageId === slicedContiguous.storageId); // true
console.log(a.id === slicedContiguous.id); // false
console.log(a.isContiguous); // true 此时是连续的

// 非连续切片
const slicedNonContiguous = a.slice("::2, ::-1"); // a[::2, ::-1]
console.log(slicedNonContiguous.shape) // [1, 3];
console.log(a.storageId === slicedNonContiguous.storageId); // true
console.log(a.id === slicedNonContiguous.id); // false
console.log(slicedNonContiguous.isContiguous); // false 此时非连续
```

#### 3️⃣ 线性代数与矩阵运算

```typescript
import * as k from '@kandle/core';

// 矩阵乘法
const x = k.randn([128, 512]);
const weight = k.randn([512, 256]);
const output = k.matmul(x, weight); // [128, 256]
console.log(output.shape);

// Batch Matrix Multiplication
const batch = k.randn([4, 64, 128]);
const weights = k.randn([4, 128, 64]);
const batchOut = k.bmm(batch, weights); // [4, 64, 64]
console.log(batchOut.shape);

// 线性层（带偏置）
const weightLinear = k.randn([256, 512]);
const bias = k.randn([256]);
const result = k.linear(x, weightLinear, bias);
console.log(result.shape);  // [128, 256]
```

#### 4️⃣ 使用 nn.Module 构建模型

```typescript
import { nn, Tensor, randn } from '@kandle/core';

class MLP extends nn.Module {
    fc1: nn.Linear;
    fc2: nn.Linear;

    constructor(inputDim: number, hiddenDim: number, outputDim: number) {
        super();
        this.fc1 = new nn.Linear(inputDim, hiddenDim);
        this.fc2 = new nn.Linear(hiddenDim, outputDim);
    }

    async forward(x: Tensor): Promise<Tensor> {
        // js无法重载, 只能单独提供call方法替代 python的 model(x)
        x = await this.fc1.call(x); 
        x = x.relu();
        x = await this.fc2.call(x);
        return x;
    }
}

// 使用模型
const model = new MLP(784, 256, 10);
const input = randn([32, 784]);
const output = await model.call(input);
console.log(output.shape);  // [32, 10]

```

#### 5️⃣ 内存管理（类似 tf.tidy）

```typescript
import * as k from '@kandle/core';

// 自动释放中间张量
const result = k.tidy( () => {
    const a = k.randn([1000, 1000]);
    const temp1 = a.mul(2);
    const temp2 = temp1.add(3);
    return temp2.sum(); // 只有 sum 结果会保留，temp1/temp2 自动释放
});

console.log('Result:', await result.dataAsync());
```

---

## 📦 Monorepo 包结构

Kandle 采用 **pnpm workspace** 组织的 Monorepo 架构，各包职责如下：

| 包名 | 功能描述 | 核心文件 |
|------|---------|---------|
| **@kandle/core** | 🎨 用户侧 API，Tensor 类、操作符、nn.Module | [src/tensor.ts](packages/core/src/tensor.ts) |
| **@kandle/backend-webgpu** | ⚡ WebGPU 后端实现（GPU 计算） | [src/index.ts](packages/backend-webgpu/src/index.ts) |
| **@kandle/types** | 📐 类型定义、接口、OpSchema | [src/opschema/](packages/types/src/opschema/) |
| **@kandle/utils** | 🛠️ 工具函数、dtype 处理、形状推断 | [src/index.ts](packages/utils/src/index.ts) |
| **@kandle/model-utils** | 🤖 模型构建工具（Qwen3、Whisper） | [src/index.ts](packages/model-utils/src/index.ts) |

---

## ✨ 核心特性

### 1. 完整的张量原语系统

#### Stride 机制与非连续内存支持
- ✅ **步长（Stride）机制**：完整实现 PyTorch 风格的内存布局管理
- ✅ **零拷贝视图操作**：`transpose`、`permute`、`slice` 等操作无需复制数据
- ✅ **非连续内存计算**：支持在 reshape、slice 后直接进行计算
- ✅ **Memory Format**：支持 Contiguous 和 ChannelsLast 布局

```typescript
// 非连续内存示例
const x = randn([4, 3, 224, 224]);
const transposed = x.transpose(1, 2); // 零拷贝，strides 改变
const sliced = x.slice("1:-1"); // 视图操作

// 自动处理非连续内存计算
const result = transposed.add(1).relu(); // 后端自动处理步长
```

#### 广播（Broadcasting）机制
完全兼容 NumPy/PyTorch 的广播规则：

```typescript
const a = randn([4, 1, 3]);
const b = randn([3]);
const result = a.add(b); // 自动广播 b 到 [4, 1, 3]
```

### 2. 丰富的 DType 支持

> 💡 **设计哲学**：逻辑 dtype 与物理 dtype 分离，后端根据设备能力自动选择存储格式

> 💡 量化类型在计划中, 以及后续会对bool / int8 / int16 / float16 做存储优化方案

| DType | TypedArray | WebGPU 存储 | 状态 | 备注 |
|-------|-----------|-------------|------|------|
| `float32` | `Float32Array` | `f32` | ✅ 完整 | 直接硬件支持 |
| `float64` | `Float64Array` | `f32` | ⚠️ 降级 | 降级为 f32，存在精度损失 |
| `float16` | `Uint16Array` | `f16` / `f32` | ⚠️ 设备相关 | 需 shader-f16 扩展 |
| `int32` | `Int32Array` | `i32` | ✅ 完整 | 直接支持 |
| `uint32` | `Uint32Array` | `u32` | ✅ 完整 | 直接支持 |
| `int8` / `uint8` | `Int8Array` / `Uint8Array` | `i32` / `u32` | ⚠️ 扩展 | 扩展存储为 32 位 |
| `int16` / `uint16` | `Int16Array` / `Uint16Array` | `i32` / `u32` | ⚠️ 降级 | 降级存储 |
| `complex64` / `complex128` | `Float32Array` / `Float64Array` | `vec2<f32>` | ⚠️ 简陋 | 交错存储 `[r0,i0,r1,i1,...]` |
| `bool` | `Uint8Array` | `u32` | ⚠️ 扩展 | 扩展存储 |


### 3. 200+ 张量操作

> 💡 列表由AI检索生成, 可能存在遗漏或者未实现的情况, 未仔细检查, 酌情参考

> 💡 以下展示为 torch 算子名称, 为了对齐javascript 开发体验, snake-case 名称会被替换为camelCase

<details>
<summary><b>📐 算术与数学运算</b></summary>

**基础算术**：`add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `abs`, `neg`, `reciprocal`, `floor`, `ceil`, `round`, `trunc`, `frac`, `sign`

**三角函数**：`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**双曲函数**：`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**指数与对数**：`exp`, `exp2`, `expm1`, `log`, `log10`, `log2`, `log1p`

**特殊函数**：`erf`, `erfc`, `sigmoid`, `logit`, `i0`

</details>

<details>
<summary><b>🔢 线性代数</b></summary>

**矩阵运算**：`matmul`, `mm`, `bmm`, `dot`, `mv`, `outer`, `addmm`, `addmv`, `baddbmm`

**矩阵操作**：`diag`, `diagonal`, `trace`, `tril`, `triu`

**分解与求解**（规划中）：`svd`, `qr`, `cholesky`, `solve`

</details>

<details>
<summary><b>🎲 归约操作</b></summary>

`sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `logsumexp`, `prod`, `norm`, `median`, `mode`, `all`, `any`

支持指定维度归约和 `keepdim` 参数：
```typescript
const x = randn([4, 5, 6]);
const result = x.sum(1, true); // 在维度 1 归约，保持维度 -> [4, 1, 6]
```

</details>

<details>
<summary><b>🔍 比较与逻辑</b></summary>

**比较运算**：`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `maximum`, `minimum`, `clamp`

**逻辑运算**：`logical_and`, `logical_or`, `logical_not`, `logical_xor`

**条件选择**：`where`, `masked_fill`, `masked_select`

</details>

<details>
<summary><b>🔀 形状操作</b></summary>

**视图操作**（零拷贝）：`view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`

**拼接与分割**：`cat`, `stack`, `split`, `chunk`, `unbind`

**索引与切片**：`slice`, `select`, `index_select`, `gather`, `scatter`, `masked_select`

**重复与扩展**：`repeat`, `repeat_interleave`, `expand`, `tile`

**翻转与旋转**：`flip`, `fliplr`, `flipud`, `rot90`, `roll`

**高级操作**：`as_strided`（直接操作 stride）

</details>

<details>
<summary><b>🧮 卷积与池化</b></summary>

**卷积**：`conv1d`, `conv2d`, `conv3d`, `conv_transpose2d`, `conv_transpose3d`

**池化**：`max_pool1d`, `max_pool2d`, `max_pool3d`, `avg_pool1d`, `avg_pool2d`, `avg_pool3d`

**自适应池化**：`adaptive_avg_pool2d`, `adaptive_max_pool2d`

**填充**：`pad`（支持 constant、reflect、replicate、circular 模式）

</details>

<details>
<summary><b>📊 归一化</b></summary>

`batch_norm`, `layer_norm`, `group_norm`, `instance_norm`, `rms_norm`, `normalize`

</details>

<details>
<summary><b>⚡ 激活函数</b></summary>

`relu`, `gelu`, `silu` (swish), `elu`, `selu`, `leaky_relu`, `prelu`, `rrelu`, `hardtanh`, `relu6`, `softplus`, `softsign`, `softmax`, `log_softmax`, `softmin`, `sigmoid`, `tanh`, `log_sigmoid`, `hardsigmoid`, `hardswish`, `mish`, `dropout`

</details>

<details>
<summary><b>🎵 FFT（快速傅里叶变换）</b></summary>

**实数 FFT**：`rfft`, `irfft`, `rfft2`, `irfft2`

**复数 FFT**：`fft`, `ifft`, `fft2`, `ifft2`

**应用场景**：音频信号处理、频谱分析

</details>

<details>
<summary><b>📈 累积操作</b></summary>

`cumsum`, `cumprod`, `cummax`, `cummin`, `diff`

</details>

<details>
<summary><b>🔧 其他实用操作</b></summary>

**排序**：`sort`, `argsort`, `topk`, `kthvalue`

**唯一值**：`unique`, `unique_consecutive`

**填充与克隆**：`fill_`, `zero_`, `clone`, `detach`

**类型转换**：`to` (dtype/device 转换), `contiguous` (强制连续内存)

</details>

### 4. 完整的 nn.Module 生态

#### 核心基类
- **`nn.Module`**：基类，支持 `forward`、`parameters()`
- **`nn.Parameter`**：可学习参数封装
- **容器**：`Sequential`, `ModuleList`, `ModuleDict`

> `state_dict()`、`load_state_dict()` 难以完全对齐, 模型加载参考下方 `IO` 类API

#### 已实现的层

<details>
<summary><b>线性层与嵌入层</b></summary>

- `nn.Linear`：全连接层
- `nn.Embedding`：嵌入层

</details>

<details>
<summary><b>卷积层</b></summary>

- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
- `nn.ConvTranspose2d`, `nn.ConvTranspose3d`

</details>

<details>
<summary><b>池化层</b></summary>

- `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.MaxPool3d`
- `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.AvgPool3d`

</details>

<details>
<summary><b>归一化层</b></summary>

- `nn.LayerNorm`
- `nn.RMSNorm`

</details>

<details>
<summary><b>激活层</b></summary>

- `nn.ReLU`, `nn.GELU`, `nn.SiLU`
- `nn.LeakyReLU`, `nn.PReLU`, `nn.Softmax`, `nn.LogSoftmax`
- `nn.Sigmoid`, `nn.Tanh`, `nn.Softplus`, `nn.Mish`

</details>

#### Hook 机制
支持前向和反向 Hook（反向需 Autograd 支持）：

```typescript
// 注册前向 Hook, register_forward_hook
model.registerForwardHook(async (module, input, output) => {
    console.log('Layer output shape:', output.shape);
});

// 前向预处理 Hook, register_forward_pre_hook
model.registerForwardPreHook(async (module, input) => {
    console.log('Layer input shape:', input.shape);
});
```

**应用场景**：
- 特征可视化（如 CAM、Grad-CAM）
- 中间层输出提取
- 模型调试与性能分析
- 动态层替换

### 5. audio 模块(对标 torchaudio)

实现了 PyTorch 音频处理库的核心功能：

<details>
<summary><b>变换（Transforms）</b></summary>

**类式 API**：
- `audio.Spectrogram`：时频谱图
- `audio.MelScale`：Mel 滤波器组
- `audio.MelSpectrogram`：Mel 频谱图
- `audio.MFCC`：梅尔倒谱系数
- `audio.AmplitudeToDB`：幅度转分贝
- `audio.InverseMelScale`：逆 Mel 变换
- `audio.GriffinLim`：相位重建
- `audio.FrequencyMasking`：频域遮罩（数据增强）
- `audio.TimeMasking`：时域遮罩（数据增强）

**函数式 API**：
对应的 `audio.functional.*` 函数

</details>

<details>
<summary><b>使用示例</b></summary>

```typescript
import { audio, Tensor } from '@kandle/core';

 // 假设有3秒的音频数据
const audioData = new Float32Array(16000 * 3); 

const waveform = new Tensor(audioData, { shape: [1, audioData.length] });

// 计算 Mel 频谱图
const melSpec = new audio.MelSpectrogram({
    sample_rate: 16000,
    n_fft: 400,
    hop_length: 160,
    n_mels: 80,
});
const melOutput = await melSpec.call(waveform); 
console.log(melOutput.shape);  // [1, 80, 301]

// 转换为对数刻度
const ampToDB = new audio.AmplitudeToDB();
const logMel = await ampToDB.call(melOutput);
console.log(logMel.shape);  // [1, 80, 301]
```

</details>

#### 6️⃣ 音频信号处理

```typescript
import { audio, Tensor } from '@kandle/core';

// 假设有3秒的音频数据
const audioData = new Float32Array(16000 * 3);

const waveform = new Tensor(audioData, { shape: [1, audioData.length] });

// 计算频谱图
const spectrogram = new audio.Spectrogram({
    n_fft: 512,
    hop_length: 256,
    power: 2.0,
});
const spec = await spectrogram.call(waveform);
console.log(spec.shape);    // [1, 257, 188]

// 应用 Mel 滤波器
const melScale = new audio.MelScale({
    n_mels: 80,
    sample_rate: 16000,
    n_stft: 257,
});
const melSpec = await melScale.call(spec);
console.log(melSpec.shape);  // [1, 80, 188]

// 计算 MFCC
const mfcc = new audio.MFCC({
    sample_rate: 16000,
    n_mfcc: 13,
    n_mels: 40
});
const mfccFeatures = await mfcc.call(waveform); 
console.log(mfccFeatures.shape); // [1, 13, 241]

// 数据增强：时域遮罩
const timeMask = new audio.TimeMasking({ time_mask_param: 10 });
const augmented = await timeMask.call(melSpec);
console.log(augmented.shape);   // [1, 80, 188]
```

### 6. I/O 系统

#### 支持的模型格式
- ✅ **Safetensor**：HuggingFace 主流格式，支持分片索引（`.safetensors.index.json`）
- ✅ **NumPy (`.npy`)**：用于测试数据加载

#### ByteSource 抽象
跨平台统一数据源接口：
- `FileByteSource`（Node.js）
- `BlobByteSource`（Web）
- `BufferByteSource`（内存）

#### Safetensor 加载示例

```typescript
import { io } from '@kandle/core';

// 加载 safetensor (只读取 header，不加载数据)
const group = await io.loadSafetensor('./model.safetensors');

// 查看所有权重
group.dumpWeightMap();

// 加载指定 tensor
const layer = group.getLayer('model.embed_tokens.weight');
const tensor = await io.tensorFromSafetensorLayer(layer!, { device: 'webgpu' });

console.log(tensor.shape, tensor.dtype);

// 释放资源
group.close();
```

完整 IO 用法见 [IO使用说明](packages/core/src/io/README.md)

### 7. Showcase：完整模型实现（对齐 PyTorch）

> 💡 **设计目标**：构造这些模型不是为了替代专用推理引擎，而是展示 Kandle 作为**白盒框架**如何实现与 PyTorch 高度对齐的模型架构。

#### 🤖 Qwen3（文本生成）

**Qwen3MLP（SwiGLU）代码对比**： HuggingFace Transformers 官方实现和 Kandle 实现

> 🐍 Python (HuggingFace Transformers)

```python
# 来源: huggingface/transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

> 📘 TypeScript (Kandle)

```typescript
// @kandle/model-utils
// src/mlp/swiglu.ts
export class SwiGLUMLP extends nn.Module {
    gate_proj: nn.Linear;
    up_proj: nn.Linear;
    down_proj: nn.Linear;

    constructor(options: SwiGLUMLPOptions) {
        super();
        const {
            hiddenSize,
            intermediateSize,
            bias = false,
        } = options;
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.gate_proj = new nn.Linear(hiddenSize, intermediateSize, bias);
        this.up_proj = new nn.Linear(hiddenSize, intermediateSize, bias);
        this.down_proj = new nn.Linear(intermediateSize, hiddenSize, bias);
        this.addModule('gate_proj', this.gate_proj);
        this.addModule('up_proj', this.up_proj);
        this.addModule('down_proj', this.down_proj);
    }
    
    async forward(x: Tensor): Promise<Tensor> {
        const gateProj = await this.gate_proj.call(x);
        const gate = functional.silu(gateProj);
        const up = await this.up_proj.call(x);
        const hidden = gate.mul(up);
        const output = await this.down_proj.call(hidden);
        return output;
    }

}
```

</td>
</tr>
</table>

> 📌 **来源说明**：Python 代码引用自 [huggingface/transformers - modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

**架构完整性**：
- ✅ `Qwen3DecoderLayer`：完整实现 Attention + MLP + LayerNorm
- ✅ `GroupedQueryAttention`：GQA with RoPE + Q/K RMSNorm
- ✅ `SwiGLUMLP`：SwiGLU 激活函数（`silu(gate) * up`）
- ✅ `nn.RMSNorm`：RMS 归一化
- ✅ 完整的前向传播流程，包括 KV Cache、Causal Mask

**完整示例**：[playground-web/qwen3/](playground-web/qwen3/)、[playground-node/src/qwen3/](playground-node/src/qwen3/)

```typescript
import { Qwen3ForCausalLM } from '@kandle/model-utils';

const model = new Qwen3ForCausalLM(config, useCausalMask = true);
await model.loadFromSafetensor(safetensorGroup);

const output = await model.forward(inputIds, {
    positionIds,
    pastKeyValues,
    attentionMask,
});
```

#### 🎤 Whisper（语音识别）
- **架构组件**：`WhisperEncoder`, `WhisperDecoder`, `WhisperModel`
- **音频处理**：集成 Mel Spectrogram 预处理
- **解码策略**：Greedy Decoding
- **完整示例**：[playground-node/src/whisper/](playground-node/src/whisper/)

```typescript
import { Whisper, prepareAudioInput } from '@kandle/model-utils';

const model = new Whisper(WHISPER_BASE_CONFIG);
await model.loadFromSafetensor(safetensorGroup);

const melInput = await prepareAudioInput(audioFloat32Array);
const result = await transcribe(model, tokenizer, melInput);
console.log(result.text);
```

#### 工具组件
- **RoPE**：`applyRotaryPosEmb`（旋转位置编码）
- **Sinusoidal 位置编码**：`sinusoidalPositionEncoding`
- **KV Cache**：`KVCache`（推理加速）
- **Attention 变体**：`multiHeadAttention`, `groupedQueryAttention`, `multiQueryAttention`
- **MLP 变体**：`SwiGLU`, `GeGLU`

---

## 🏗️ 架构设计

### 分层架构图

```
┌─────────────────────────────────────────────────────────┐
│          User API Layer (@kandle/core)                  │
│  Tensor, zeros, randn, nn.Module, audio...              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               Dispatch Layer                            │
│  Operation routing, dtype resolution, broadcasting      │
└────────────────────┬────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼──────┐  ┌────▼──────┐  ┌────▼──────┐
│ Handler 1 │  │ Handler 2 │  │ Handler N │  (Mechanism-based)
│ Map/Reduce│  │ Composite │  │   FFT     │
└────┬──────┘  └────┬──────┘  └────┬──────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Kernel Layer                            │
│  Backend-specific implementations                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
┌────────▼─────────┐   ┌─────────▼──────────┐
│ @kandle/backend- │   │ @kandle/backend-js │
│      webgpu      │   │   (CPU fallback)   │
└──────────────────┘   └────────────────────┘
```

### 核心概念

#### 存算分离（Storage & Handle）
参考 PyTorch 的 ATen/c10 设计：

```typescript
// 1. Storage：物理内存
interface IStorage {
    data: TypedArray;
    byteOffset: number;
    byteLength: number;
}

// 2. TensorHandle：元数据
interface ITensorHandle {
    storage: IStorage;
    shape: number[];
    strides: number[];
    offset: number;
    dtype: DType;
}

// 3. Tensor：用户侧封装
class Tensor {
    constructor(public handle: ITensorHandle) {}
    
    // 视图操作只修改 handle，不复制 storage
    transpose(dim0: number, dim1: number): Tensor {
        const newStrides = swapStrides(this.handle.strides, dim0, dim1);
        return new Tensor({ ...this.handle, strides: newStrides });
    }
}
```

**优势**：
- ✅ 零拷贝视图操作
- ✅ 支持非连续内存布局
- ✅ 灵活的内存管理策略

#### Dispatch 系统（简化版分发机制）

> ⚠️ **与 PyTorch 的差异**：PyTorch 使用复杂的 Dispatch Key 系统（如 `AutogradCPU`、`AutogradCUDA`），支持多维度分发（backend、layout、autograd）。Kandle 当前实现的是**简化版本**，基于 `opName + device` 进行分发。

> 📝 **架构演进**：当前的 dispatch 路由机制会在后续版本中重写，但核心的机制化路由思想不会改变。

按**计算机制**分类路由：

```typescript
// packages/utils/src/dispatchUtils.ts
const handlers = {
    'map_reduce': MapReduceHandler,     // 元素级运算 + 归约
    'composite': CompositeHandler,      // 纯 JS 组合操作
    'fft': FFTHandler,                  // FFT 专用处理
    'conv': ConvolutionHandler,         // 卷积专用
    'matmul': MatmulHandler,            // 矩阵乘法专用
    ....
};

// 简化的分发逻辑（非 Dispatch Key）
function dispatch(opSchema: OpSchema, ...args) {
    const handler = handlers[opSchema.mechanism];
    const backend = getBackendByDevice(args[0].device);
    return handler.execute(backend, opSchema, ...args);
}
```

**当前实现**：
- ✅ 按 `mechanism` 字段路由到不同 Handler
- ✅ 按 `device` 获取对应 Backend（webgpu / js）
- ❌ 不支持 PyTorch 风格的多维度 Dispatch Key
- ❌ 不支持运行时动态注册 Dispatch 规则 (开发中)

#### DType Resolver（逻辑与物理分离）
自动处理 dtype 转换和设备兼容性：

```typescript
// 用户代码
const x = randn([100], { dtype: 'float64' });

// 后端实际存储（WebGPU 不支持 f64）
// 逻辑 dtype: float64
// 物理 dtype: float32（降级）
// 上传时：Float64Array -> Float32Array（精度损失警告）
// 下载时：Float32Array -> Float64Array
```

**特性**：
- 自动检测 `shader-f16` 扩展
- 透明处理 dtype 降级
- 支持复数类型的 `vec2<f32>` 映射

#### Codegen 系统（参考 PyTorch native_functions.yaml）

> 💡 **设计灵感**：PyTorch 使用 `native_functions.yaml` 定义算子签名，通过 torchgen 生成 C++ 代码。Kandle 实现了类似思路，使用 **TypeScript Interface** 作为 OpSchema，通过 Codegen 生成用户侧 API。

**生成器**：[文件位置](scripts/codegen)

**生成文件**：[文件位置](packages/core/src/generated)

减少手写代码，确保 API 一致性：

```bash
pnpm codegen
```

**OpSchema 定义示例**：

```typescript
// packages/types/src/opschema/ops/activation.ts
export const gelu: OpEntry = {
    name: 'gelu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'approximate', type: SchemaT.String(['none', 'tanh']), default: 'none' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['approximate'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'gelu',
    codegen: { tensorMethod: 'gelu', namespace: 'nn.functional' },
};
```

**生成内容**：
- `methods-gen.ts`：Tensor 原型方法（如 `tensor.add()`）
- `ops-gen.ts`：顶层操作函数（如 `add(tensor, other)`）
- `types-gen.ts`：OpSchema 类型定义汇总

**对比 PyTorch**：

| 特性 | PyTorch (YAML) | Kandle (TypeScript Interface) |
|------|---------------|-------------------------------|
| **定义格式** | `native_functions.yaml` | TypeScript Interface |
| **生成目标** | C++ / Python Binding | TypeScript API |
| **类型检查** | 运行时 | 编译时（TypeScript） |
| **扩展性** | ✅ 支持复杂 Dispatch | ⚠️ 当前简化版 |

---

## 🎯 特殊处理

### 1. Python 风格的 Slice 语法

```typescript
import { randn, slice } from '@kandle/core';

const x = randn([3, 4, 5]);
// Python: x[:, 1:5, ::2]
// Kandle:
const result = x.slice(":,1:5,::2");
 console.log(result.shape); // [3,3,3]

// 支持负索引
const tail = x.slice("-5:"); // x[-5:]
console.log(tail.shape);    // [3,4,5]
```


---

## ⚠️ 已知限制与问题

> 详细文档见 [knownIssues/](knownIssues/)

### 1. 异步传染（Async Propagation）
**问题**：WebGPU 的 `buffer.mapAsync()` 强制所有数据读取为异步  
**影响**：
- ✅ `forward` 方法统一为 `async`
- ❌ 无法在 kernel 中直接读取其他 Tensor 的值（如条件判断）
- ❌ 组合算子实现复杂度提升

**缓解措施**：
- 提供同步的 JS 后端（开发中）
- 设计上避免需要同步读取的操作

**详细说明**：[knownIssues/async.md](knownIssues/async.md)

### 2. DType 降级
**问题**：WebGPU 不支持部分 dtype，需降级或扩展存储  
**影响**：
- `float64` → `float32`：精度损失
- `int8` → `i32`：内存浪费 4 倍
- `complex128` → `vec2<f32>`：精度损失

**建议**：
- 优先使用 `float32` 和 `int32`
- 需要高精度时使用 JS 后端(开发中)

**详细说明**：见 [核心特性 - DType 支持](#2-丰富的-dtype-支持)

### 3. 复数支持简陋
**问题**：当前复数类型实现较为基础，仅支持基本算术  
**规划**：后续版本将重构复数运算系统

**详细说明**：[knownIssues/complex.md](knownIssues/complex.md)

### 4. 类型系统待加强
**问题**：存在大量 `as any` 类型断言  
**规划**：逐步加强 TypeScript 类型推断和泛型约束

**详细说明**：[knownIssues/type.md](knownIssues/type.md)

### 5. Dispatch 层职责混合
**问题**：当前 dispatch 层混合了调度逻辑和部分计算逻辑  
**规划**：重构为纯粹的路由层

**详细说明**：[knownIssues/dispatch.md](knownIssues/dispatch.md)、[knownIssues/opschema.md](knownIssues/opschema.md)

### 6. WebGPU 数值稳定性问题

**问题**：WebGPU 后端在不同硬件/驱动下可能产生数值差异，特别是在某些激活函数（如 GELU、softmax）和数学运算中可能出现 NaN 或精度问题

**影响**：
- ⚠️ 在不同 GPU 设备上，相同模型的输出可能存在微小差异
- ❌ 极端情况下可能产生 NaN 值（如未 clamp 的 GELU、exp 溢出的 softmax）
- 🔴 由硬件/驱动实现差异导致的数值不稳定似乎无法完全避免?

**已知案例**：
- **GELU 激活函数 NaN**：未限制 tanh 输入范围时，在某些层的大激活值下会产生 NaN（详见 [knownIssues/shader.md](knownIssues/shader.md)）
- **Softmax 溢出**：输入未减去 max 值时，exp 可能溢出产生 Infinity
- **精度损失累积**：多层计算后，float32 精度损失可能累积

**缓解措施**：
- ✅ 已对关键算子添加数值稳定性保护（如 GELU 添加 clamp，softmax 减去 max）
- ⚠️ 使用相同硬件进行测试和部署，避免跨设备结果差异
- 📊 对关键输出进行数值范围监控，及时发现异常
- 🔍 参考 [knownIssues/shader.md](knownIssues/shader.md) 了解详细的排查指南

**当前限制**：
- 由于 WebGPU 规范未强制要求精确的浮点运算行为，不同驱动/硬件的实现可能存在差异
- 目前没有特别好的解决方案来完全消除这种差异，这是 WebGPU 生态的固有限制

**详细说明**：[knownIssues/shader.md](knownIssues/shader.md)

### 7. WebGPU 显存泄漏与内存管理

**问题**：WebGPU 后端存在显存泄漏问题，这是由于 JavaScript 侧无法感知 WebGPU 侧的内存压力导致的

**根本原因**：
- ❌ **JS 与 WebGPU 内存隔离**：JavaScript 的垃圾回收机制（GC）无法感知 GPU 显存压力
- ❌ **FinalizationRegistry 时机不可控**：即使使用 `FinalizationRegistry` 注册析构函数，回调触发时机完全由 GC 决定，可能在显存已耗尽后才触发
- ⚠️ **View Tensor 引用复杂**：`transpose`、`slice` 等操作创建的视图 Tensor 与原 Tensor 共享 Storage，引用关系复杂，难以精确判断释放时机

**影响**：
- ❌ 长时间推理（如生成 1000+ tokens）可能因显存耗尽而崩溃
- ⚠️ 大模型加载后，即使不再使用的中间 Tensor 也可能占用显存
- ⚠️ View 操作（如 `view()`, `transpose()`）虽然不复制数据，但会延长原 Storage 的生命周期

**我的优化尝试**：
- ⚠️ 实现了一套复杂的 Memory Pool 机制，复用 GPU Buffer, 但是没有取得实际效果, 因此当前发布版本未启用 见 [文件位置](packages/backend-webgpu/src/memory-pool).
- ✅ 提供了 `tidy()` 和手动 `dispose()` API
- ✅ 尝试优化 View Tensor 的引用计数
- ⚠️ **但依然存在问题**：由于 JS/WebGPU 内存隔离的本质限制，无法做到完美的自动管理

**缓解措施**（需用户配合）：
- **强烈推荐**：使用 `tidy()` 包裹计算逻辑，自动管理中间 Tensor 生命周期
  ```typescript
  const result = tidy(() => {
      const temp1 = a.mul(2);
      const temp2 = temp1.add(3);
      return temp2.sum(); // 只有 sum 结果会保留
  });
  ```
- 显式调用 `dispose()` 释放不再使用的 Tensor
  ```typescript
  const temp = a.mul(2);
  const result = temp.add(3);
  temp.dispose(); // 手动释放
  ```
- 定期监控显存使用情况（Chrome DevTools → Performance Monitor）
- 避免在循环中创建大量临时 Tensor 而不释放

**长期规划**：
- 优化 Memory Pool 策略，更激进的内存回收
- 改进 View Tensor 的引用追踪机制

**希望得到高人指点!**

**详细说明**：[knownIssues/cache.md](knownIssues/cache.md)

---

## 🌐 浏览器兼容性

### WebGPU 支持情况

| 浏览器 | 最低版本 | 备注 |
|--------|---------|------|
| Chrome | 113+ | ✅ 完整支持 |
| Edge | 113+ | ✅ 完整支持 |
| Safari | 预览版 | ⚠️ 部分支持（macOS 14+）|
| Firefox | 实验性 | ⚠️ 需手动启用 |

---

## 📚 示例项目

### Web 环境：Qwen3 文本生成
**位置**：[playground-web/qwen3/](playground-web/qwen3/)

```bash
cd playground-web
pnpm install
pnpm dev
# 访问 http://localhost:5173/qwen3/
```

**功能**：
- WebGPU 加速的文本生成
- 支持流式输出
- 可视化 Attention 权重

### Node.js 环境：Whisper 语音识别
**位置**：[playground-node/src/whisper/](playground-node/src/whisper/)

```bash
cd playground-node
pnpm install
pnpm start
```

**功能**：
- 加载本地音频文件
- Mel Spectrogram 预处理
- 端到端语音转文字

---

## 🚀 开发路线

### 🔨 开发中（当前版本）

- **架构重构**：进一步优化分层设计，完善 Codegen 系统和类型推断
- **Autograd（自动微分）**：反向传播系统，支持梯度计算和参数优化
  - 当前正在实现基于 `derivatives.yaml` 的自动求导系统
  - 完全参照 PyTorch 的 DSL 设计，实现 TypeScript 版本的解析器 (有些复杂, 实际上借助AI直接实现所有原语算子可能更快)
  - 通过 derivatives.yaml 自动生成反向传播算子，确保与 PyTorch 行为一致
  - 目标：覆盖大部分常用前向算子的梯度定义，支持高阶导数
- **nn.Module 增强**：
  - ✅ Generator 实现的逐层调试
  - 🚧 动态层替换（Runtime Module Swapping）
  - 🚧 状态检查点（Checkpoint）
- **Custom Kernel 注册**：运行时注册自定义 kernel，支持 Fused Kernel 优化
- **纯 JS 后端完善**：完全同步的 CPU 计算后端（类比 PyTorch CPU）
- **领域模块的完善**: 继续完善audio模块(对标torchaudio), 以及vision 模块(对标 torchvision)

### 📅 近期规划（3-6 个月）

- **量化支持**：
  - `int4`, `int8` 量化 dtype
  - 动态量化（Dynamic Quantization）
  - 静态量化（Static Quantization）
- **独立标量数学库**：解决 JS 下混合 dtype 计算的类型转换问题
- **性能优化**：
  - Kernel Fusion
  - Memory Pool 优化
  - Shader 缓存系统

### 🌟 长期规划（6-12 个月）

- **远程后端**：基于 WebSocket/gRPC 的分布式计算后端
- **Training API**：完整的训练循环支持（需 Autograd 完成）
- **NumPy API 兼容层**：复用计算分发架构, 加入 `numpy` 算子, 以 namespace 形式暴露 `import { np } from '@kandle/core'`
- **模型可解释性 UI 组件库**（基于 React）：
  - 热力图可视化（Heatmap）
  - 特征图展示（Feature Maps）
  - Attention 权重可视化
  - 推理过程动画
- **预训练模型生态**：
  - 推出独立的 `@kandle/models` package，实现类似 HuggingFace Transformers 的功能
  - 提供开箱即用的预训练模型（LLaMA、BERT、ViT、Whisper 等）
  - 支持直接从 HuggingFace Hub 加载模型和配置
  - 统一的模型加载和推理接口
- **GitHub Agent 自动化工作流**：
  - 实现智能 GitHub Agent，监听特定格式的 Issue/PR 请求
  - 当匹配到算子需求时，自动拉起 Agent 执行以下流程：
    1. 搜索相关技术资料和 PyTorch 实现
    2. 生成算子定义（OpSchema）
    3. 实现 Kernel（WebGPU/JS 双后端）
    4. 自动生成功能测试和数值验证用例
    5. 提交 Pull Request 供人工审核
  - 降低社区贡献门槛，加速算子生态建设

---

## 🎭 API 设计原则

### 代码风格说明

> ⚠️ **命名规范过渡中**：由于 Vibe Coding 的客观原因，当前代码中存在 `snake_case` 和 `camelCase` 混用的情况。我会在后续版本中逐步统一过渡到 `camelCase`，以对齐 JavaScript/TypeScript 社区的习惯。

### JavaScript 本地化的妥协

由于 JavaScript 和 Python 语言特性差异，部分 API 无法完全对齐：

#### 1. 参数命名
**Python（关键字参数）**：
```python
torch.zeros(size=(3, 4), dtype=torch.float32, device='cuda')
```

**JavaScript（对象参数）**：
```typescript
zeros([3, 4], { dtype: 'float32', device: 'webgpu' })
```

#### 2. 运算符重载
由于 JavaScript 不支持运算符重载，基础运算需要显式调用方法：

| Python | TypeScript (Kandle) |
|--------|--------------------|
| `a + b` | `add(a, b)` 或 `a.add(b)` |
| `a - b` | `sub(a, b)` 或 `a.sub(b)` |
| `a * b` | `mul(a, b)` 或 `a.mul(b)` |
| `a / b` | `div(a, b)` 或 `a.div(b)` |
| `a @ b` | `matmul(a, b)` 或 `a.matmul(b)` |
| `model(x)` | `model.call(x)` |

> 💡 `nn.Module` 的 `__call__` 需要显式调用 `.call()` 方法

#### 3. 切片语法
**Python**：
```python
x[:, 1:5]
```

**JavaScript（函数模拟）**：
```typescript
x.slice(":,1:5")
```

### 后续版本的 API 演进
关于参数位置,2个方案2选1:
- **完全对齐torch**: 通过复杂的重载, 尝试完全对齐.
> 大部分API可行, 但实现过于复杂, 且少数API会无法对齐, 需要单独记忆, 体验不一致
- **设计JS规范**: 设计一套js的对标规范, 通过规则约束, 保证"转译后对齐"
> 开发简单, 但会造成体验下降和torch对齐程度降低

---

## ⚡ 性能说明

### 设计权衡

Kandle 采用 **Eager Mode**（动态图）执行，这与静态图推理引擎存在本质差异：

| 特性 | Eager Mode (Kandle) | Static Graph (ONNX) |
|------|---------------------|---------------------|
| **执行方式** | 逐操作执行 | 一次性图优化 |
| **中间状态** | ✅ 随时可访问 | ❌ 编译后不可见 |
| **动态控制流** | ✅ 支持 if/loop | ⚠️ 受限 |
| **内存开销** | ⚠️ 高（保留中间结果） | ✅ 优化后较低 |
| **推理速度** | ⚠️ 较慢（无全局优化） | ✅ 极致优化 |
| **调试体验** | ✅ 优秀 | ❌ 困难 |

### 适用场景

**✅ 推荐使用 Kandle**：
- 研究与原型开发
- 模型调试与可解释性分析
- 需要中间计算的应用（如音频预处理 + 模型推理）
- 教学与学习

**❌ 不推荐使用 Kandle**：
- 生产环境的高性能推理（请使用 **ONNX Runtime**）
- 移动端/边缘设备（请使用 **WebLLM** 或 TFLite）
- 对延迟极度敏感的实时应用

### 性能优化建议

- **避免不必要的数据读取**：减少 `dataAsync()` 调用
- **使用 `tidy()` 管理内存**：自动释放中间张量
- **Batch 推理**：增大 batch size 提升 GPU 利用率

---

## 🤖 关于 AI 辅助开发

### Vibe Coding 实践与探索

> 💡 **这同时也是一次 Vibe Coding 极限的探索尝试**

本项目采用 **Vibe Coding** 开发模式，试图探索 AI 辅助开发的边界：
- **架构设计**：由本人负责（研读 PyTorch ATen/c10 源码）
- **代码实现**：主要由 AI 辅助完成（Gemini、Claude）
- **测试与验证**：人类 + AI 协同（NumPy/PyTorch 参考测试）

### Vibe Coding 的挑战

在这个项目中，我尝试让 AI 完成：
- ✅ **200+ 算子实现**：从数学公式到 WebGPU Shader 代码
- ✅ **复杂架构落地**：Stride 机制、Dispatch 系统、Autograd（开发中）
- ✅ **跨平台适配**：WebGPU / 纯 JS 双后端
- ⚠️ **边界条件处理**：当前仍是短板，需人工介入

### 代码质量说明

由于模型幻觉和 Vibe Coding 的客观原因：
- ⚠️ 代码风格未完全统一（会在后期重构）
- ⚠️ 部分注释可能存在不一致或过时
- ⚠️ 边界情况处理不完善（快乐路径优先）
- ⚠️ 核心逻辑经过测试验证, 但覆dtype盖率不够, 且部分算子未进行数值稳定性测试（参考 PyTorch/NumPy）

### 开发效率提升

通过 AI 辅助，实现了：
- 📈 **10x+ 开发速度**：200+ 操作符在数周内完成
- 🔄 **快速迭代**：多次架构重构（从 v1 到 v11）
- 📚 **自动化文档**：README、API 文档、设计文档
- 🧪 **测试用例生成**：自动对齐 PyTorch 行为

### Vibe Coding 的局限

- ❌ **架构决策**：仍需人类深度思考
- ❌ **性能优化**：AI 难以理解显存布局、Cache 优化等细节
- ❌ **调试复杂问题**：非连续内存、类型推断等需人工介入
- ❌ **长期一致性**：跨文件重构时容易引入不一致

**经验总结**：Vibe Coding 适合"有明确规范的重复性工作"（如算子实现），但核心架构设计仍需人类主导。

---

## 💭 写在最后的一些话

### 为什么要做 Kandle？

起初，我只是想在 js 环境下使用 onnxruntime 做推理。onnxruntime 做推理时，需要处理大量的中间张量，这在 JavaScript 下简直是种折磨。
原生 JS 临时写的一些方法只能处理特定的"一次性"流程。比如，你很难通用地处理一个高维数组的切片、视图变换或是复杂的广播操作，然后再在其他模型的推理过程中复用这个方法。

### "你为什么不用 tfjs? "

由于历史原因，学术界和前沿模型大多采用 PyTorch 范式构建。在 JavaScript 生态中，缺乏对应的 API 支持，在复现论文或移植模型时会面临巨大的思维转换成本.

### "你为什么不用 Python？"

因为我不爱写 Python，甚至说得上是讨厌。虽然 Python 的 AI 生态由于历史原因处于垄断地位，但对于习惯了 C-like 风格语言开发，以及长期使用强类型系统的人来说，Python 下的开发是折磨的。我很难对 `def` / `None` / 使用"游标卡尺"这些特性感到顺手。特别是万恶的 `**kwargs`，你真的知道正在写什么吗？

最初，我尝试将预处理动作（比如音频的 Mel Spec 计算）直接导出为 ONNX 图。但很快我发现这不可行，生态的碎片化会导致模型导出极其繁琐。例如你要推理 Whisper，不同版本的预处理参数存在微小但致命的差异（如 turbo-v3 的 mel_spec n_mels 是 128，而 base 版本是 80），这意味着我需要为每种情况导出特定的模型版本。当我尝试在 js 下推理越多的模型，这种"生态鸿沟"就越明显。

### "你为什么不用 transformers.js？"

我当然试过 transformers.js，它很好，开箱即用, 支持很多主流模型. 但同时也正是因为这一点, 它有一个核心问题：它基于 onnxruntime, **是一个黑盒**。你只能通过有限的参数去调整 Pipeline，你很难掌控数据流转的细节。如果你想要深度定制或优化流程，这让人非常抓狂。

### 我们缺少的是什么？

冷静思考后我发现，我痛苦的根源本身，不是 JS 下缺少机器学习框架——事实上已经有 tfjs，tfjs 很强大，但它的 API 设计理念源于上一代深度学习框架。当我想在 JS 里随手写一个 x.view().transpose() 时，我发现我不得不去查阅那些对我来说略显陌生的文档。它很好，但它不是我习惯的那个‘标准’.

我们也不缺推理框架，onnxruntime / WebLLM 都已经做了深度的优化。

**我们缺少 PyTorch**，或者说缺少一个事实上的、符合 Torch 标准的 API。

自然地，我搜索到了 torch-js 等现有库。虽然它们对 ATen 和 c10（PyTorch 的核心 C++ 库）进行了大量的 Binding 工作，但遗憾的是，它们大多并未完成所有工作，且许多项目在多年前就停止了维护。这无疑让人更沮丧了。

真的没有办法了吗？真的只能回去写 Python 吗？

### AI 带来的可能性

好消息是，这么多年我也还一直在写 C++，更好的消息是，在这个时代，我们有了"外挂"——AI。

我clone了 torch的仓库, 深入阅读了 ATen 和 c10 的源码。在 Gemini 的辅助下，我大概读懂了它们的设计：计算分发系统（Dispatch）、代码生成系统 、存储与计算分离的设计哲学等等。这套架构非常强大，但也很复杂。

然后，我试着在 TypeScript 下复刻一个简化的版本。

### 无数次推倒重来的旅程

这一路是无数个 MVP 版本的推倒重来：

- 从最初仅能计算 `T + T` 的简陋版本，逐步实现了类型提升和标量计算
- 从混乱的数据类型管理，到清晰区分逻辑 Dtype 与物理 Dtype
- 从必须操作连续内存，到学会了计算 Stride（步长），实现了内存视图与维度折叠
- 再到实现广播、高级索引
- 最后完成存算分离、后端隔离以及用户侧友好的 API 设计……

终于，我确信我掌握了张量计算的所有必要拼图。

接下来的事情就变得纯粹了：写文档，设计 prompt，然后指挥 AI 把代码落地。

开发过程中并非一帆风顺。每当少漏掉一层抽象，或者没有考虑到某些关键设计. 往往就意味着随后的大规模重构。当然，还要和 Vibe Coding 模式下特有的"模型幻觉"作斗争。即使是 vibe coding，这也很痛苦。

但是，只要一想到 **"如果我不做这个，那就要回去写 Python"** , 你就会能量无限

然后，就有了现在的 Kandle。

### 但是我还是得说明

**关于 Autograd**：最初为了聚焦推理场景并减少工程量，我砍掉了Autograd。事实上，我也很难想象在 JS 下进行大规模模型训练的场景。但对于反向传播的思路，本质上与前向传播共享同一套分发系统。但既然已经做到这个程度了，不做 Autograd 就太可惜了，抱着这种心理，最终还是决定补上 Autograd。

**关于成熟度**：现在的 Kandle 依然是一个"玩具"，最多是一个"精致的玩具"。因为核心是 eager mode，所以更多的应用场景在中间计算/预处理/后处理，和状态检查等。除非目标模型无法导出 onnx 或者 transformers.js 没有实现 pipeline，否则如果直接落地到具体的推理业务，性能会远低于专门的推理框架。

但同样的 eager mode 也有黑盒们无法替代的优势。**现在你能完全地"解剖"模型，每一层，每一次前向传播你都能完整地掌控**。我同样也在 nn.Module 中设计了 hook，比如现在可以和 DOM API 结合使用，现在我们有更炫酷、更直观的方式进行模型可解释性分析。在新的版本中我尝试使用了 generator/yield 重写了 nn.Module，现在可以"移交"传播时的控制权, 你完全可以随时"暂停"某一层的计算，像在 IDE 里打断点那样进行 debug。

### 未来的可能性

我也还有很多想法暂时没有实现。比如由于解耦的彻底，现在我能实现**远程后端**，通过 gRPC/WebSocket 等方案进行交互，就像在 Web 下调用 WebGPU 那样，用户侧下发计算，只在"同步点"进行阻塞获取数据。这在理论上完全可行。

我个人认为 **Torch 这套设计完全能够成为一套"张量计算协议"标准**，而不仅仅只是 Python 下的机器学习框架，完全还能做到更多。

当然，以我的能力现在只能做到这里了。等我"进化"之后，我会试着更进一步。

---

## 📄 许可证

MIT License

---

<div align="center">

**🌟 如果这个项目对你有帮助，请给个 Star！**

**💬 如果你对"JavaScript 版 PyTorch"有任何想法，欢迎在 Issues/Discussions 分享**

Made with ❤️ by Vibe Coding

</div>

---