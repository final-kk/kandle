<div align="center">

# 🕯️ Kandle

**JavaScript Native PyTorch-aligned Machine Learning Framework**

**[中文版 (Chinese Version)](./README.zh-CN.md)**

*Bringing the true PyTorch experience to the JavaScript ecosystem*

[Quick Start](#-quick-start) • [Core Features](#-core-features) • [Example Projects](#-example-projects) • [Architecture](#-architecture-design) • [Roadmap](#-roadmap)

</div>

---

## 📖 Introduction

Kandle is a **JavaScript Native** machine learning framework that adopts an **Eager Mode** (dynamic graph) execution pattern, deeply referencing PyTorch's ATen/c10 architectural design. I view PyTorch not just as a Python framework, but as the **API specification standard** for modern AI frameworks. Kandle is dedicated to implementing an API system highly aligned with PyTorch within the JavaScript ecosystem.

### 🎯 Core Value Proposition

* **🔄 Dynamic Graph Execution**: True Eager Mode, supporting layer-by-layer debugging, intermediate state inspection, and dynamic control flow.
* **🎨 PyTorch API Alignment**: Aligned at the architectural level rather than simple API wrapping, reducing migration costs and learning curves.
* **⚡ Hybrid Backend Architecture**: Supports both WebGPU (GPU acceleration) and pure JS (CPU computation) backends under a unified interface.
* **🧩 Complete Tensor System**: Implements a full Stride mechanism, broadcasting, view operations, and non-contiguous memory support.
* **🎵 Rich Operator Library**: 200+ tensor operations covering arithmetic, linear algebra, convolution, FFT, audio processing, and more.
* **🚀 Out-of-the-Box Models**: Native support for mainstream models like Qwen3 and Whisper, capable of loading Safetensor weights directly.

### 💡 Why Choose Kandle?

Current inference engines in the JavaScript ecosystem, such as ONNX Runtime and WebLLM, are excellent but are fundamentally **Blackbox Systems** focused on static graph inference. Kandle, as a **Whitebox Framework**, fills the following gaps:

| Requirement | Blackbox Inference Engines | Kandle (Whitebox Framework) |
| --- | --- | --- |
| **Intermediate Computation** | ❌ Cannot intervene after static graph compilation | ✅ Pause/Inspect at any layer via dynamic graph |
| **Model Interpretability** | ❌ Blackbox, internal states inaccessible | ✅ Hooks, layer-by-layer state export |
| **Custom Compute Flow** | ❌ Limited to predefined Pipelines | ✅ Fully programmable control flow |
| **Pre/Post-processing** | ⚠️ Requires extra toolchains / ONNX export | ✅ Unified tensor operation system |
| **API Learning Curve** | ⚠️ Framework-proprietary APIs | ✅ Zero cost for PyTorch users |
| **Debugging Experience** | ❌ Hard to pinpoint issues in a blackbox | ✅ "Breakpoint-style" step-by-step debugging |
| **Inference Performance** | ✅ Static graph global optimization | ⚠️ Eager Mode trade-off |

**What Whitebox can do that Blackbox cannot**:

* 🔬 **Layer-wise Feature Extraction**: Export intermediate Tensors at any layer for visual analysis.
* 🎨 **Runtime Layer Replacement**: Dynamically replace/skip certain layers to implement model pruning or A/B testing.
* 🧪 **Custom Loss Functions**: Design special computation paths combined with business logic.
* 🎯 **Precise Memory Control**: Manually manage Tensor lifecycles to optimize VRAM usage.
* 🌐 **Deep Integration with DOM API**: Hooks directly bind to Canvas/WebGL for real-time rendering.

**Suitable Scenarios**: Research, prototype development, model debugging, applications requiring intermediate calculations, audio/visual pre-processing, interpretability analysis.
**Unsuitable Scenarios**: High-performance production inference (please use ONNX Runtime or WebLLM).

---

## 🚨 Technical Verification Prototype Disclaimer

> ⚠️ **This is a technical verification prototype, not a production-ready preview.**

* ✅ The current version focuses on **Forward Propagation Architecture Verification**, implementing 200+ operators and a complete nn.Module system.
* 🚧 **Autograd (Backpropagation)** is under development and will be fully implemented in the next version.
* ⚠️ **Happy Path Disclaimer**: The current implementation mainly verifies the main flow (Happy Path); edge cases and error handling are not yet perfect.
* 🔒 **No PRs Accepted Yet**: The current development branch has completely diverged from the public version with breaking changes. Contributions will be opened after the architecture stabilizes.
* 💬 **Feedback Welcome**: I have been working somewhat in isolation, so I am very eager to hear the community's thoughts and suggestions on "what a JavaScript version of PyTorch should look like."
* 🎯 **Operator Demand Collection**: Besides primitive operators, I want to know which specific operators the community needs supported early on.

---

## 🌐 Online Experience

No installation required, experience Kandle immediately. We provide a visual interactive Demo based on **Qwen3-0.6B**, fully showcasing the unique advantages of an Eager Mode framework in **Model Interpretability**:

### 📍 Access Addresses

* **🤗 HuggingFace Spaces**: [https://huggingface.co/spaces/finalkk/kandle-demo](https://huggingface.co/spaces/finalkk/kandle-demo)
* **⚡ Vercel**: [http://kandle-demo.vercel.app/](http://kandle-demo.vercel.app/)

### ✨ Demo Core Features

| Feature | Description |
| --- | --- |
| **🎯 Step-by-Step Execution** | Execute forward propagation step by step |
| **⏮️ Time Travel** | Step back and re-select the generation path |
| **🎲 Manual Intervention** | Manually select candidate words at each token generation to explore different branches |
| **🔍 Logit Lens** | Visualize the probability distribution of each layer's output in the vocabulary space |
| **🔗 Attention Links** | Interactively view Self-Attention weight connection relationships |
| **🔥 Heatmap Visualization** | Real-time display of Attention Maps and activation value distributions |

> 💡 **This is the meaning of a Whitebox framework**: Not just reasoning, but "dissecting" every step of the calculation process.

### 🎬 Usage Suggestions

1. **Explore the Model's Thought Process**: Observe the top-k tokens of each layer's output during single-step execution to understand how the model gradually "focuses" on the final answer.
2. **Compare Different Paths**: Backtrack and select different candidate words to observe the bifurcation points of the generation results.
3. **Discover Attention Patterns**: Use Attention Links to discover key tokens the model focuses on (e.g., pronoun resolution, context dependencies).
4. **Debugging and Teaching**: Suitable for researchers to understand the internal mechanisms of Transformers, or for teaching demonstrations.

### ⚠️ Demo Limitations

1. **Original Pre-trained Version Only**: Currently, techniques like quantization are not implemented; it only loads original bf16 weights.
2. **Relatively Large Model Size**: The original model size is about 1.5GB. It is recommended to download the model manually and load it using WebFile or Upload.
[Qwen3-0.6B Link](https://huggingface.co/Qwen/Qwen3-0.6B)

---

## 🚀 Quick Start

### Installation

```bash
# Browser environment only needs the core library
# Using pnpm (Recommended)
pnpm add @kandle/core @kandle/backend-webgpu

# Optional type library, utilities, and pre-model building tools
pnpm add @kandle/types @kandle/utils @kandle/model-utils

# Or using npm
npm install @kandle/core @kandle/backend-webgpu

# If running in a Node.js environment, install webgpu polyfill additionally
npm install webgpu

```

### Environment Requirements

* **Node.js**: ≥ 18.0.0 (ES2020+ support required)
* **Browser**: Chrome/Edge ≥ 113 (WebGPU support)
* **TypeScript**: ≥ 5.0 (Optional)

### Basic Usage Examples

#### 1️⃣ Initialize Backend (WebGPU)

```typescript
import { env } from "@kandle/core";
import { WebGPUBackend } from "@kandle/backend-webgpu";

export async function initWebGPU() {
    const backend = await WebGPUBackend.create();
    env.setBackend(backend);
    env.setDefaultDevice(backend.name);
}

```

#### 2️⃣ Tensor Operations and Broadcasting

```typescript
import * as k from '@kandle/core';
import { Tensor } from '@kandle/core';

// Create Tensor
const a = new Tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'float32' });
const b = k.randn([2, 3]);

// Arithmetic operations (supports broadcasting)
const result = a.add(b).mul(2).softmax(-1);

// Get data (WebGPU asynchronous read)
const data = await result.dataAsync();
console.log(data); // Float32Array [...]

// Shape operations (Zero-copy views)
const transposed = a.transpose(0, 1);
console.log(transposed.shape); // [3, 2]
console.log(a.storageId === transposed.storageId); // true
console.log(a.id === transposed.id); // false
const reshaped = a.reshape([3, 2]);
console.log(reshaped.shape); // [3, 2]
console.log(a.storageId === reshaped.storageId); // true
console.log(a.id === reshaped.id); // false


// Advanced Indexing (Python style)
const slicedContiguous = a.slice(":1, 1:"); // a[:1, 1:]
console.log(slicedContiguous.shape) // [1, 2];
console.log(a.storageId === slicedContiguous.storageId); // true
console.log(a.id === slicedContiguous.id); // false
console.log(a.isContiguous); // true (contiguous here)

// Non-contiguous slicing
const slicedNonContiguous = a.slice("::2, ::-1"); // a[::2, ::-1]
console.log(slicedNonContiguous.shape) // [1, 3];
console.log(a.storageId === slicedNonContiguous.storageId); // true
console.log(a.id === slicedNonContiguous.id); // false
console.log(slicedNonContiguous.isContiguous); // false (non-contiguous here)

```

#### 3️⃣ Linear Algebra and Matrix Operations

```typescript
import * as k from '@kandle/core';

// Matrix Multiplication
const x = k.randn([128, 512]);
const weight = k.randn([512, 256]);
const output = k.matmul(x, weight); // [128, 256]
console.log(output.shape);

// Batch Matrix Multiplication
const batch = k.randn([4, 64, 128]);
const weights = k.randn([4, 128, 64]);
const batchOut = k.bmm(batch, weights); // [4, 64, 64]
console.log(batchOut.shape);

// Linear Layer (with bias)
const weightLinear = k.randn([256, 512]);
const bias = k.randn([256]);
const result = k.linear(x, weightLinear, bias);
console.log(result.shape);  // [128, 256]

```

#### 4️⃣ Building Models with nn.Module

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
        // JS cannot overload operators, must provide call method to replace Python's model(x)
        x = await this.fc1.call(x); 
        x = x.relu();
        x = await this.fc2.call(x);
        return x;
    }
}

// Using the model
const model = new MLP(784, 256, 10);
const input = randn([32, 784]);
const output = await model.call(input);
console.log(output.shape);  // [32, 10]

```

#### 5️⃣ Memory Management (Like tf.tidy)

```typescript
import * as k from '@kandle/core';

// Automatically release intermediate tensors
const result = k.tidy( () => {
    const a = k.randn([1000, 1000]);
    const temp1 = a.mul(2);
    const temp2 = temp1.add(3);
    return temp2.sum(); // Only the sum result is kept, temp1/temp2 are automatically released
});

console.log('Result:', await result.dataAsync());

```

---

## 📦 Monorepo Package Structure

Kandle uses a Monorepo architecture organized by **pnpm workspace**. The responsibilities of each package are as follows:

| Package Name | Function Description | Core File |
| --- | --- | --- |
| **@kandle/core** | 🎨 User-side API, Tensor class, Operators, nn.Module | [src/tensor.ts](packages/core/src/tensor.ts) |
| **@kandle/backend-webgpu** | ⚡ WebGPU Backend Implementation (GPU Compute) | [src/index.ts](packages/backend-webgpu/src/index.ts) |
| **@kandle/types** | 📐 Type definitions, Interfaces, OpSchema | [src/opschema/](packages/types/src/opschema/) |
| **@kandle/utils** | 🛠️ Utility functions, dtype handling, shape inference | [src/index.ts](packages/utils/src/index.ts) |
| **@kandle/model-utils** | 🤖 Model building tools (Qwen3, Whisper) | [src/index.ts](packages/model-utils/src/index.ts) |

---

## ✨ Core Features

### 1. Complete Tensor Primitive System

#### Stride Mechanism & Non-Contiguous Memory Support

* ✅ **Stride Mechanism**: Fully implements PyTorch-style memory layout management.
* ✅ **Zero-Copy View Operations**: Operations like `transpose`, `permute`, `slice` do not copy data.
* ✅ **Non-Contiguous Memory Computation**: Supports direct computation after reshape or slice.
* ✅ **Memory Format**: Supports Contiguous and ChannelsLast layouts.

```typescript
// Non-contiguous memory example
const x = randn([4, 3, 224, 224]);
const transposed = x.transpose(1, 2); // Zero-copy, strides changed
const sliced = x.slice("1:-1"); // View operation

// Automatically handles non-contiguous memory computation
const result = transposed.add(1).relu(); // Backend handles strides automatically

```

#### Broadcasting Mechanism

Fully compatible with NumPy/PyTorch broadcasting rules:

```typescript
const a = randn([4, 1, 3]);
const b = randn([3]);
const result = a.add(b); // Automatically broadcasts b to [4, 1, 3]

```

### 2. Rich DType Support

> 💡 **Design Philosophy**: Logical dtype is separated from physical dtype; the backend automatically selects storage format based on device capabilities.

> 💡 Quantized types are planned, and storage optimization for bool / int8 / int16 / float16 will be added later.

| DType | TypedArray | WebGPU Storage | Status | Notes |
| --- | --- | --- | --- | --- |
| `float32` | `Float32Array` | `f32` | ✅ Full | Direct hardware support |
| `float64` | `Float64Array` | `f32` | ⚠️ Downgrade | Downgrades to f32, precision loss exists |
| `float16` | `Uint16Array` | `f16` / `f32` | ⚠️ Device Dependent | Requires shader-f16 extension |
| `int32` | `Int32Array` | `i32` | ✅ Full | Direct support |
| `uint32` | `Uint32Array` | `u32` | ✅ Full | Direct support |
| `int8` / `uint8` | `Int8Array` / `Uint8Array` | `i32` / `u32` | ⚠️ Extended | Extended storage to 32-bit |
| `int16` / `uint16` | `Int16Array` / `Uint16Array` | `i32` / `u32` | ⚠️ Downgrade | Downgraded storage |
| `complex64` / `complex128` | `Float32Array` / `Float64Array` | `vec2<f32>` | ⚠️ Rudimentary | Interleaved storage `[r0,i0,r1,i1,...]` |
| `bool` | `Uint8Array` | `u32` | ⚠️ Extended | Extended storage |

### 3. 200+ Tensor Operations

> 💡 List generated by AI retrieval, may contain omissions or unimplemented items. Please refer with caution.

> 💡 The following shows torch operator names. To align with JavaScript development experience, snake-case names are replaced with camelCase.

<details>
<summary><b>📐 Arithmetic & Math Operations</b></summary>

**Basic Arithmetic**: `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `abs`, `neg`, `reciprocal`, `floor`, `ceil`, `round`, `trunc`, `frac`, `sign`

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Exponential & Logarithmic**: `exp`, `exp2`, `expm1`, `log`, `log10`, `log2`, `log1p`

**Special Functions**: `erf`, `erfc`, `sigmoid`, `logit`, `i0`

</details>

<details>
<summary><b>🔢 Linear Algebra</b></summary>

**Matrix Operations**: `matmul`, `mm`, `bmm`, `dot`, `mv`, `outer`, `addmm`, `addmv`, `baddbmm`

**Matrix Manipulation**: `diag`, `diagonal`, `trace`, `tril`, `triu`

**Decomposition & Solving** (Planned): `svd`, `qr`, `cholesky`, `solve`

</details>

<details>
<summary><b>🎲 Reduction Operations</b></summary>

`sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `logsumexp`, `prod`, `norm`, `median`, `mode`, `all`, `any`

Supports reduction on specific dimensions and `keepdim` parameter:

```typescript
const x = randn([4, 5, 6]);
const result = x.sum(1, true); // Reduce on dim 1, keep dim -> [4, 1, 6]

```

</details>

<details>
<summary><b>🔍 Comparison & Logic</b></summary>

**Comparison**: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `maximum`, `minimum`, `clamp`

**Logic**: `logical_and`, `logical_or`, `logical_not`, `logical_xor`

**Conditional Selection**: `where`, `masked_fill`, `masked_select`

</details>

<details>
<summary><b>🔀 Shape Operations</b></summary>

**View Operations** (Zero Copy): `view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`

**Concatenation & Splitting**: `cat`, `stack`, `split`, `chunk`, `unbind`

**Indexing & Slicing**: `slice`, `select`, `index_select`, `gather`, `scatter`, `masked_select`

**Repetition & Expansion**: `repeat`, `repeat_interleave`, `expand`, `tile`

**Flipping & Rotating**: `flip`, `fliplr`, `flipud`, `rot90`, `roll`

**Advanced**: `as_strided` (Direct stride manipulation)

</details>

<details>
<summary><b>🧮 Convolution & Pooling</b></summary>

**Convolution**: `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d`, `conv_transpose3d`

**Pooling**: `max_pool1d`, `max_pool2d`, `max_pool3d`, `avg_pool1d`, `avg_pool2d`, `avg_pool3d`

**Adaptive Pooling**: `adaptive_avg_pool2d`, `adaptive_max_pool2d`

**Padding**: `pad` (Supports constant, reflect, replicate, circular modes)

</details>

<details>
<summary><b>📊 Normalization</b></summary>

`batch_norm`, `layer_norm`, `group_norm`, `instance_norm`, `rms_norm`, `normalize`

</details>

<details>
<summary><b>⚡ Activation Functions</b></summary>

`relu`, `gelu`, `silu` (swish), `elu`, `selu`, `leaky_relu`, `prelu`, `rrelu`, `hardtanh`, `relu6`, `softplus`, `softsign`, `softmax`, `log_softmax`, `softmin`, `sigmoid`, `tanh`, `log_sigmoid`, `hardsigmoid`, `hardswish`, `mish`, `dropout`

</details>

<details>
<summary><b>🎵 FFT (Fast Fourier Transform)</b></summary>

**Real FFT**: `rfft`, `irfft`, `rfft2`, `irfft2`

**Complex FFT**: `fft`, `ifft`, `fft2`, `ifft2`

**Application**: Audio signal processing, spectrum analysis

</details>

<details>
<summary><b>📈 Cumulative Operations</b></summary>

`cumsum`, `cumprod`, `cummax`, `cummin`, `diff`

</details>

<details>
<summary><b>🔧 Other Utilities</b></summary>

**Sorting**: `sort`, `argsort`, `topk`, `kthvalue`

**Unique Values**: `unique`, `unique_consecutive`

**Filling & Cloning**: `fill_`, `zero_`, `clone`, `detach`

**Type Conversion**: `to` (dtype/device conversion), `contiguous` (force contiguous memory)

</details>

### 4. Complete nn.Module Ecosystem

#### Core Base Classes

* **`nn.Module`**: Base class, supports `forward`, `parameters()`
* **`nn.Parameter`**: Learnable parameter wrapper
* **Containers**: `Sequential`, `ModuleList`, `ModuleDict`

> `state_dict()` and `load_state_dict()` are hard to align perfectly, refer to the `IO` class API below for model loading.

#### Implemented Layers

<details>
<summary><b>Linear & Embedding Layers</b></summary>

* `nn.Linear`: Fully connected layer
* `nn.Embedding`: Embedding layer

</details>

<details>
<summary><b>Convolution Layers</b></summary>

* `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
* `nn.ConvTranspose2d`, `nn.ConvTranspose3d`

</details>

<details>
<summary><b>Pooling Layers</b></summary>

* `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.MaxPool3d`
* `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.AvgPool3d`

</details>

<details>
<summary><b>Normalization Layers</b></summary>

* `nn.LayerNorm`
* `nn.RMSNorm`

</details>

<details>
<summary><b>Activation Layers</b></summary>

* `nn.ReLU`, `nn.GELU`, `nn.SiLU`
* `nn.LeakyReLU`, `nn.PReLU`, `nn.Softmax`, `nn.LogSoftmax`
* `nn.Sigmoid`, `nn.Tanh`, `nn.Softplus`, `nn.Mish`

</details>

#### Hook Mechanism

Supports Forward and Backward Hooks (Backward requires Autograd support):

```typescript
// Register forward Hook, register_forward_hook
model.registerForwardHook(async (module, input, output) => {
    console.log('Layer output shape:', output.shape);
});

// Register forward pre-hook, register_forward_pre_hook
model.registerForwardPreHook(async (module, input) => {
    console.log('Layer input shape:', input.shape);
});

```

**Use Cases**:

* Feature Visualization (e.g., CAM, Grad-CAM)
* Intermediate Layer Output Extraction
* Model Debugging and Profiling
* Dynamic Layer Replacement

### 5. audio Module (benchmarking torchaudio)

Implements core functionality of PyTorch's audio processing library:

<details>
<summary><b>Transforms</b></summary>

**Class API**:

* `audio.Spectrogram`: Spectrogram
* `audio.MelScale`: Mel Filter Bank
* `audio.MelSpectrogram`: Mel Spectrogram
* `audio.MFCC`: Mel-frequency cepstral coefficients
* `audio.AmplitudeToDB`: Amplitude to Decibels
* `audio.InverseMelScale`: Inverse Mel Transform
* `audio.GriffinLim`: Phase Reconstruction
* `audio.FrequencyMasking`: Frequency Masking (Data Augmentation)
* `audio.TimeMasking`: Time Masking (Data Augmentation)

**Functional API**:
Corresponding `audio.functional.*` functions

</details>

<details>
<summary><b>Usage Example</b></summary>

```typescript
import { audio, Tensor } from '@kandle/core';

 // Assume 3 seconds of audio data
const audioData = new Float32Array(16000 * 3); 

const waveform = new Tensor(audioData, { shape: [1, audioData.length] });

// Compute Mel Spectrogram
const melSpec = new audio.MelSpectrogram({
    sample_rate: 16000,
    n_fft: 400,
    hop_length: 160,
    n_mels: 80,
});
const melOutput = await melSpec.call(waveform); 
console.log(melOutput.shape);  // [1, 80, 301]

// Convert to log scale
const ampToDB = new audio.AmplitudeToDB();
const logMel = await ampToDB.call(melOutput);
console.log(logMel.shape);  // [1, 80, 301]

```

</details>

#### 6️⃣ Audio Signal Processing

```typescript
import { audio, Tensor } from '@kandle/core';

// Assume 3 seconds of audio data
const audioData = new Float32Array(16000 * 3);

const waveform = new Tensor(audioData, { shape: [1, audioData.length] });

// Compute Spectrogram
const spectrogram = new audio.Spectrogram({
    n_fft: 512,
    hop_length: 256,
    power: 2.0,
});
const spec = await spectrogram.call(waveform);
console.log(spec.shape);    // [1, 257, 188]

// Apply Mel Filter
const melScale = new audio.MelScale({
    n_mels: 80,
    sample_rate: 16000,
    n_stft: 257,
});
const melSpec = await melScale.call(spec);
console.log(melSpec.shape);  // [1, 80, 188]

// Compute MFCC
const mfcc = new audio.MFCC({
    sample_rate: 16000,
    n_mfcc: 13,
    n_mels: 40
});
const mfccFeatures = await mfcc.call(waveform); 
console.log(mfccFeatures.shape); // [1, 13, 241]

// Data Augmentation: Time Masking
const timeMask = new audio.TimeMasking({ time_mask_param: 10 });
const augmented = await timeMask.call(melSpec);
console.log(augmented.shape);   // [1, 80, 188]

```

### 6. I/O System

#### Supported Model Formats

* ✅ **Safetensor**: HuggingFace mainstream format, supports shard index (`.safetensors.index.json`)
* ✅ **NumPy (`.npy`)**: Used for test data loading

#### ByteSource Abstraction

Unified data source interface across platforms:

* `FileByteSource` (Node.js)
* `BlobByteSource` (Web)
* `BufferByteSource` (Memory)

#### Safetensor Loading Example

```typescript
import { io } from '@kandle/core';

// Load safetensor (read header only, data not loaded)
const group = await io.loadSafetensor('./model.safetensors');

// View all weights
group.dumpWeightMap();

// Load specific tensor
const layer = group.getLayer('model.embed_tokens.weight');
const tensor = await io.tensorFromSafetensorLayer(layer!, { device: 'webgpu' });

console.log(tensor.shape, tensor.dtype);

// Release resources
group.close();

```

Full IO usage see [IO Documentation](packages/core/src/io/README.md)

### 7. Showcase: Full Model Implementation (Aligned with PyTorch)

> 💡 **Design Goal**: Constructing these models is not to replace dedicated inference engines, but to demonstrate how Kandle, as a **Whitebox Framework**, implements model architectures highly aligned with PyTorch.

#### 🤖 Qwen3 (Text Generation)

**Qwen3MLP (SwiGLU) Code Comparison**: HuggingFace Transformers Official vs. Kandle Implementation

> 🐍 Python (HuggingFace Transformers)

```python
# Source: huggingface/transformers
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

> 📌 **Source Note**: Python code referenced from [huggingface/transformers - modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

**Architecture Completeness**:

* ✅ `Qwen3DecoderLayer`: Fully implements Attention + MLP + LayerNorm
* ✅ `GroupedQueryAttention`: GQA with RoPE + Q/K RMSNorm
* ✅ `SwiGLUMLP`: SwiGLU activation (`silu(gate) * up`)
* ✅ `nn.RMSNorm`: RMS Normalization
* ✅ Complete Forward Propagation flow, including KV Cache, Causal Mask

**Full Example**: [playground-web/qwen3/](playground-web/qwen3/), [playground-node/src/qwen3/](playground-node/src/qwen3/)

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

#### 🎤 Whisper (Speech Recognition)

* **Architecture Components**: `WhisperEncoder`, `WhisperDecoder`, `WhisperModel`
* **Audio Processing**: Integrated Mel Spectrogram preprocessing
* **Decoding Strategy**: Greedy Decoding
* **Full Example**: [playground-node/src/whisper/](playground-node/src/whisper/)

```typescript
import { Whisper, prepareAudioInput } from '@kandle/model-utils';

const model = new Whisper(WHISPER_BASE_CONFIG);
await model.loadFromSafetensor(safetensorGroup);

const melInput = await prepareAudioInput(audioFloat32Array);
const result = await transcribe(model, tokenizer, melInput);
console.log(result.text);

```

#### Utility Components

* **RoPE**: `applyRotaryPosEmb`
* **Sinusoidal Positional Encoding**: `sinusoidalPositionEncoding`
* **KV Cache**: `KVCache` (Inference acceleration)
* **Attention Variants**: `multiHeadAttention`, `groupedQueryAttention`, `multiQueryAttention`
* **MLP Variants**: `SwiGLU`, `GeGLU`

---

## 🏗️ Architecture Design

### Layered Architecture Diagram

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

### Core Concepts

#### Separation of Storage & Computation (Storage & Handle)

Referencing PyTorch's ATen/c10 design:

```typescript
// 1. Storage: Physical memory
interface IStorage {
    data: TypedArray;
    byteOffset: number;
    byteLength: number;
}

// 2. TensorHandle: Metadata
interface ITensorHandle {
    storage: IStorage;
    shape: number[];
    strides: number[];
    offset: number;
    dtype: DType;
}

// 3. Tensor: User-side wrapper
class Tensor {
    constructor(public handle: ITensorHandle) {}
    
    // View operations modify handle only, no storage copy
    transpose(dim0: number, dim1: number): Tensor {
        const newStrides = swapStrides(this.handle.strides, dim0, dim1);
        return new Tensor({ ...this.handle, strides: newStrides });
    }
}

```

**Advantages**:

* ✅ Zero-copy view operations
* ✅ Supports non-contiguous memory layouts
* ✅ Flexible memory management strategies

#### Dispatch System (Simplified Distribution Mechanism)

> ⚠️ **Difference from PyTorch**: PyTorch uses a complex Dispatch Key system (e.g., `AutogradCPU`, `AutogradCUDA`) supporting multi-dimensional dispatch (backend, layout, autograd). Kandle currently implements a **simplified version** based on `opName + device` dispatch.

> 📝 **Architecture Evolution**: The current dispatch routing mechanism will be rewritten in future versions, but the core mechanized routing philosophy remains.

Routing by **Computation Mechanism**:

```typescript
// packages/utils/src/dispatchUtils.ts
const handlers = {
    'map_reduce': MapReduceHandler,     // Element-wise + Reduction
    'composite': CompositeHandler,      // Pure JS composite operations
    'fft': FFTHandler,                  // FFT specialized processing
    'conv': ConvolutionHandler,         // Convolution specialized
    'matmul': MatmulHandler,            // Matrix Multiplication specialized
    ....
};

// Simplified dispatch logic (Non-Dispatch Key)
function dispatch(opSchema: OpSchema, ...args) {
    const handler = handlers[opSchema.mechanism];
    const backend = getBackendByDevice(args[0].device);
    return handler.execute(backend, opSchema, ...args);
}

```

**Current Implementation**:

* ✅ Route to different Handlers by `mechanism` field
* ✅ Get corresponding Backend (webgpu / js) by `device`
* ❌ Does not support PyTorch-style multi-dimensional Dispatch Key
* ❌ Does not support runtime dynamic registration of Dispatch rules (Under development)

#### DType Resolver (Logical vs. Physical Separation)

Automatically handles dtype conversion and device compatibility:

```typescript
// User code
const x = randn([100], { dtype: 'float64' });

// Backend actual storage (WebGPU does not support f64)
// Logical dtype: float64
// Physical dtype: float32 (downgrade)
// Upload: Float64Array -> Float32Array (precision loss warning)
// Download: Float32Array -> Float64Array

```

**Features**:

* Auto-detects `shader-f16` extension
* Transparently handles dtype downgrading
* Supports `vec2<f32>` mapping for complex types

#### Codegen System (Reference PyTorch native_functions.yaml)

> 💡 **Design Inspiration**: PyTorch uses `native_functions.yaml` to define operator signatures and generates C++ code via torchgen. Kandle implements a similar idea, using **TypeScript Interface** as OpSchema and generating user-side APIs via Codegen.

**Generator**: [File Location](scripts/codegen)

**Generated Files**: [File Location](packages/core/src/generated)

Reduces boilerplate and ensures API consistency:

```bash
pnpm codegen

```

**OpSchema Definition Example**:

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

**Generated Content**:

* `methods-gen.ts`: Tensor prototype methods (e.g., `tensor.add()`)
* `ops-gen.ts`: Top-level operation functions (e.g., `add(tensor, other)`)
* `types-gen.ts`: OpSchema type definition summary

**Comparison with PyTorch**:

| Feature | PyTorch (YAML) | Kandle (TypeScript Interface) |
| --- | --- | --- |
| **Definition Format** | `native_functions.yaml` | TypeScript Interface |
| **Generation Target** | C++ / Python Binding | TypeScript API |
| **Type Check** | Runtime | Compile-time (TypeScript) |
| **Extensibility** | ✅ Supports Complex Dispatch | ⚠️ Current Simplified Version |

---

## 🎯 Special Handling

### 1. Python-style Slice Syntax

```typescript
import { randn, slice } from '@kandle/core';

const x = randn([3, 4, 5]);
// Python: x[:, 1:5, ::2]
// Kandle:
const result = x.slice(":,1:5,::2");
 console.log(result.shape); // [3,3,3]

// Supports negative indexing
const tail = x.slice("-5:"); // x[-5:]
console.log(tail.shape);    // [3,4,5]

```

---

## ⚠️ Known Limits and Issues

> Detailed documentation see [knownIssues/](knownIssues/)

### 1. Async Propagation

**Issue**: WebGPU's `buffer.mapAsync()` forces all data reading to be asynchronous.
**Impact**:

* ✅ `forward` method is unified as `async`.
* ❌ Cannot directly read values of other Tensors in kernel (e.g., conditional judgment).
* ❌ Complexity of implementing composite operators increases.

**Mitigation**:

* Provide synchronous JS backend (Under development).
* Design to avoid operations requiring synchronous reading.

**Details**: [knownIssues/async.md](knownIssues/async.md)

### 2. DType Downgrading

**Issue**: WebGPU does not support some dtypes, requiring downgrading or extended storage.
**Impact**:

* `float64` → `float32`: Precision loss.
* `int8` → `i32`: Memory waste (4x).
* `complex128` → `vec2<f32>`: Precision loss.

**Recommendation**:

* Prioritize `float32` and `int32`.
* Use JS backend for high precision (Under development).

**Details**: See [Core Features - DType Support](#-2-rich-dtype-support)

### 3. Rudimentary Complex Support

**Issue**: Current complex type implementation is basic, only supporting basic arithmetic.
**Plan**: Will refactor the complex number calculation system in future versions.

**Details**: [knownIssues/complex.md](knownIssues/complex.md)

### 4. Type System Needs Strengthening

**Issue**: Significant use of `as any` type assertions.
**Plan**: Gradually strengthen TypeScript type inference and generic constraints.

**Details**: [knownIssues/type.md](knownIssues/type.md)

### 5. Dispatch Layer Responsibility Mix

**Issue**: The current dispatch layer mixes scheduling logic with some computation logic.
**Plan**: Refactor into a pure routing layer.

**Details**: [knownIssues/dispatch.md](knownIssues/dispatch.md), [knownIssues/opschema.md](knownIssues/opschema.md)

### 6. WebGPU Numerical Stability Issues

**Issue**: WebGPU backend may produce numerical differences across different hardware/drivers, especially in certain activation functions (like GELU, softmax) and mathematical operations, leading to NaN or precision issues.

**Impact**:

* ⚠️ Identical models may produce slightly different outputs on different GPU devices.
* ❌ Extreme cases may produce NaN values (e.g., unclamped GELU, softmax exp overflow).
* 🔴 Numerical instability caused by hardware/driver implementation differences seems unavoidable?

**Known Cases**:

* **GELU Activation NaN**: Without limiting tanh input range, large activation values in certain layers can produce NaN (See [knownIssues/shader.md](knownIssues/shader.md)).
* **Softmax Overflow**: If input is not subtracted by max value, exp may overflow to Infinity.
* **Precision Loss Accumulation**: Float32 precision loss may accumulate after multi-layer computation.

**Mitigation**:

* ✅ Numerical stability protection added to key operators (e.g., clamp for GELU, subtract max for softmax).
* ⚠️ Use identical hardware for testing and deployment to avoid cross-device result differences.
* 📊 Monitor numerical ranges of key outputs to detect anomalies in time.
* 🔍 Refer to [knownIssues/shader.md](knownIssues/shader.md) for detailed troubleshooting guides.

**Current Limitations**:

* Since the WebGPU specification does not mandate precise floating-point behavior, implementations across drivers/hardware may vary.
* There is currently no excellent solution to completely eliminate this difference; this is an inherent limitation of the WebGPU ecosystem.

**Details**: [knownIssues/shader.md](knownIssues/shader.md)

### 7. WebGPU VRAM Leaks and Memory Management

**Issue**: The WebGPU backend suffers from VRAM leaks because the JavaScript side cannot perceive WebGPU side memory pressure.

**Root Causes**:

* ❌ **JS & WebGPU Memory Isolation**: JavaScript's Garbage Collection (GC) mechanism cannot perceive GPU VRAM pressure.
* ❌ **FinalizationRegistry Timing Uncontrollable**: Even using `FinalizationRegistry` to register destructors, the callback trigger timing is entirely decided by GC and may trigger after VRAM is exhausted.
* ⚠️ **Complex View Tensor References**: View Tensors created by `transpose`, `slice`, etc., share Storage with the original Tensor, creating complex reference relationships that make precise release timing difficult to determine.

**Impact**:

* ❌ Long inference sessions (e.g., generating 1000+ tokens) may crash due to VRAM exhaustion.
* ⚠️ Even after loading large models, intermediate Tensors that are no longer used may still occupy VRAM.
* ⚠️ View operations (like `view()`, `transpose()`) extend the lifecycle of the original Storage even though they don't copy data.

**My Optimization Attempts**:

* ⚠️ Implemented a complex Memory Pool mechanism to reuse GPU Buffers, but it didn't achieve practical results, so it is disabled in the current release. See [File Location](packages/backend-webgpu/src/memory-pool).
* ✅ Provided `tidy()` and manual `dispose()` APIs.
* ✅ Attempted to optimize reference counting for View Tensors.
* ⚠️ **But problems persist**: Due to the inherent limitation of JS/WebGPU memory isolation, perfect automatic management is impossible.

**Mitigation** (User Cooperation Required):

* **Highly Recommended**: Use `tidy()` to wrap computation logic to automatically manage intermediate Tensor lifecycles.
```typescript
const result = tidy(() => {
    const temp1 = a.mul(2);
    const temp2 = temp1.add(3);
    return temp2.sum(); // Only the sum result is kept
});

```


* Explicitly call `dispose()` to release unused Tensors.
```typescript
const temp = a.mul(2);
const result = temp.add(3);
temp.dispose(); // Manual release

```


* Periodically monitor VRAM usage (Chrome DevTools → Performance Monitor).
* Avoid creating massive temporary Tensors in loops without releasing them.

**Long-term Plan**:

* Optimize Memory Pool strategy for more aggressive memory reclamation.
* Improve reference tracking mechanism for View Tensors.

**Looking for expert advice!**

**Details**: [knownIssues/cache.md](knownIssues/cache.md)

---

## 🌐 Browser Compatibility

### WebGPU Support Status

| Browser | Minimum Version | Notes |
| --- | --- | --- |
| Chrome | 113+ | ✅ Full Support |
| Edge | 113+ | ✅ Full Support |
| Safari | Preview | ⚠️ Partial Support (macOS 14+) |
| Firefox | Experimental | ⚠️ Requires Manual Enable |

---

## 📚 Example Projects

### Web Environment: Qwen3 Text Generation

**Location**: [playground-web/qwen3/](playground-web/qwen3/)

```bash
cd playground-web
pnpm install
pnpm dev
# Access http://localhost:5173/qwen3/

```

**Features**:

* WebGPU accelerated text generation
* Streaming output support
* Visualized Attention weights

### Node.js Environment: Whisper Speech Recognition

**Location**: [playground-node/src/whisper/](playground-node/src/whisper/)

```bash
cd playground-node
pnpm install
pnpm start

```

**Features**:

* Loads local audio files
* Mel Spectrogram preprocessing
* End-to-end speech-to-text

---

## 🚀 Roadmap

### 🔨 In Development (Current Version)

* **Architecture Refactoring**: Further optimize layered design, refine Codegen system and type inference.
* **Autograd (Automatic Differentiation)**: Backpropagation system supporting gradient calculation and parameter optimization.
* Currently implementing an auto-differentiation system based on `derivatives.yaml`.
* Designing a TypeScript version of the parser referencing PyTorch's DSL (Complex, AI might implement all primitive operators faster).
* Automatically generate backpropagation operators via derivatives.yaml, ensuring consistency with PyTorch behavior.
* Goal: Cover gradient definitions for most common forward operators, support higher-order derivatives.


* **nn.Module Enhancements**:
* ✅ Generator-implemented layer-by-layer debugging.
* 🚧 Runtime Module Swapping.
* 🚧 State Checkpoints.


* **Custom Kernel Registration**: Runtime custom kernel registration, supporting Fused Kernel optimization.
* **Pure JS Backend Completion**: Fully synchronous CPU computation backend (analogous to PyTorch CPU).
* **Domain Module Completion**: Continue perfecting the audio module (benchmarking torchaudio) and vision module (benchmarking torchvision).

### 📅 Short-term Plan (3-6 Months)

* **Quantization Support**:
* `int4`, `int8` quantization dtypes.
* Dynamic Quantization.
* Static Quantization.


* **Independent Scalar Math Library**: Solve type conversion issues for mixed dtype calculations in JS.
* **Performance Optimization**:
* Kernel Fusion.
* Memory Pool Optimization.
* Shader Cache System.



### 🌟 Long-term Plan (6-12 Months)

* **Remote Backend**: Distributed computing backend based on WebSocket/gRPC.
* **Training API**: Complete training loop support (requires Autograd completion).
* **NumPy API Compatibility Layer**: Reuse computation dispatch architecture, add `numpy` operators, exposed via namespace `import { np } from '@kandle/core'`.
* **Model Interpretability UI Component Library** (React-based):
* Heatmap Visualization.
* Feature Maps display.
* Attention Weight Visualization.
* Inference Process Animation.


* **Pre-trained Model Ecosystem**:
* Launch independent `@kandle/models` package, implementing functionality similar to HuggingFace Transformers.
* Provide out-of-the-box pre-trained models (LLaMA, BERT, ViT, Whisper, etc.).
* Support loading models and configs directly from HuggingFace Hub.
* Unified model loading and inference interface.


* **GitHub Agent Automated Workflow**:
* Implement intelligent GitHub Agent listening for specific Issue/PR formats.
* When matching operator requests, automatically trigger Agent to:
1. Search relevant technical docs and PyTorch implementations.
2. Generate operator definitions (OpSchema).
3. Implement Kernel (WebGPU/JS dual backend).
4. Automatically generate functional tests and numerical validation cases.
5. Submit Pull Request for human review.


* Lower community contribution threshold and accelerate operator ecosystem construction.



---

## 🎭 API Design Principles

### Code Style Note

> ⚠️ **Naming Convention Transition**: Due to objective reasons related to Vibe Coding, the current code contains a mix of `snake_case` and `camelCase`. I will gradually unify this to `camelCase` in future versions to align with JavaScript/TypeScript community habits.

### Compromises for JavaScript Localization

Due to language differences between JavaScript and Python, some APIs cannot be perfectly aligned:

#### 1. Parameter Naming

**Python (Keyword Arguments)**:

```python
torch.zeros(size=(3, 4), dtype=torch.float32, device='cuda')

```

**JavaScript (Object Arguments)**:

```typescript
zeros([3, 4], { dtype: 'float32', device: 'webgpu' })

```

#### 2. Operator Overloading

Since JavaScript does not support operator overloading, basic operations require explicit method calls:

| Python | TypeScript (Kandle) |
| --- | --- |
| `a + b` | `add(a, b)` or `a.add(b)` |
| `a - b` | `sub(a, b)` or `a.sub(b)` |
| `a * b` | `mul(a, b)` or `a.mul(b)` |
| `a / b` | `div(a, b)` or `a.div(b)` |
| `a @ b` | `matmul(a, b)` or `a.matmul(b)` |
| `model(x)` | `model.call(x)` |

> 💡 `nn.Module`'s `__call__` needs to be explicitly called via `.call()` method.

#### 3. Slicing Syntax

**Python**:

```python
x[:, 1:5]

```

**JavaScript (Function Simulation)**:

```typescript
x.slice(":,1:5")

```

### API Evolution in Future Versions

Regarding parameter positioning, two options are considered:

* **Full Alignment with Torch**: Attempt complete alignment via complex overloading.

> Most APIs feasible, but implementation is overly complex, and a few APIs will fail to align, requiring separate memorization, leading to inconsistent experience.

* **Design JS Specification**: Design a JS benchmark specification, enforcing "alignment after translation" via rules.

> Simpler development, but leads to degraded experience and lower alignment with Torch.

---

## ⚡ Performance

### Design Trade-offs

Kandle uses **Eager Mode** (dynamic graph) execution, which differs fundamentally from static graph inference engines:

| Feature | Eager Mode (Kandle) | Static Graph (ONNX) |
| --- | --- | --- |
| **Execution Style** | Op-by-Op execution | One-time graph optimization |
| **Intermediate State** | ✅ Accessible anytime | ❌ Invisible after compilation |
| **Dynamic Control Flow** | ✅ Supports if/loop | ⚠️ Limited |
| **Memory Overhead** | ⚠️ High (keeps intermediate results) | ✅ Low after optimization |
| **Inference Speed** | ⚠️ Slower (no global optimization) | ✅ Extreme optimization |
| **Debugging Experience** | ✅ Excellent | ❌ Difficult |

### Applicable Scenarios

**✅ Recommend Kandle**:

* Research and Prototype Development
* Model Debugging and Interpretability Analysis
* Applications requiring intermediate calculations (e.g., Audio Preprocessing + Model Inference)
* Teaching and Learning

**❌ Do Not Recommend Kandle**:

* High-performance production inference (Please use **ONNX Runtime**)
* Mobile/Edge devices (Please use **WebLLM** or TFLite)
* Real-time applications strictly sensitive to latency

### Performance Optimization Suggestions

* **Avoid Unnecessary Data Reads**: Reduce `dataAsync()` calls.
* **Use `tidy()` for Memory**: Automatically release intermediate tensors.
* **Batch Inference**: Increase batch size to improve GPU utilization.

---

## 🤖 About AI Assisted Development

### Vibe Coding Practice and Exploration

> 💡 **This is also an exploration of the limits of Vibe Coding.**

This project adopts the **Vibe Coding** development mode, attempting to explore the boundaries of AI-assisted development:

* **Architecture Design**: Responsible by myself (Reading PyTorch ATen/c10 source code).
* **Code Implementation**: Mainly assisted by AI (Gemini, Claude).
* **Testing & Verification**: Human + AI collaboration (NumPy/PyTorch reference tests).

### Challenges of Vibe Coding

In this project, I tried to let AI complete:

* ✅ **200+ Operator Implementations**: From mathematical formulas to WebGPU Shader code.
* ✅ **Complex Architecture Landing**: Stride mechanism, Dispatch system, Autograd (In progress).
* ✅ **Cross-platform Adaptation**: WebGPU / Pure JS dual backend.
* ⚠️ **Edge Case Handling**: Currently a shortcoming, requires human intervention.

### Code Quality Note

Due to model hallucinations and objective reasons of Vibe Coding:

* ⚠️ Code style is not fully unified (will be refactored later).
* ⚠️ Some comments may be inconsistent or outdated.
* ⚠️ Edge case handling is imperfect (Happy Path priority).
* ⚠️ Core logic is verified, but dtype coverage is insufficient, and some operators lack numerical stability tests (Reference PyTorch/NumPy).

### Development Efficiency Boost

With AI assistance, achieved:

* 📈 **10x+ Development Speed**: 200+ operators completed in weeks.
* 🔄 **Fast Iteration**: Multiple architecture refactors (from v1 to v11).
* 📚 **Automated Documentation**: README, API Docs, Design Docs.
* 🧪 **Test Case Generation**: Automatically align PyTorch behavior.

### Limitations of Vibe Coding

* ❌ **Architecture Decisions**: Still requires deep human thinking.
* ❌ **Performance Optimization**: AI struggles to understand details like memory layout, Cache optimization.
* ❌ **Debugging Complex Issues**: Non-contiguous memory, type inference, etc., require human intervention.
* ❌ **Long-term Consistency**: Easy to introduce inconsistencies during cross-file refactoring.

**Summary**: Vibe Coding is suitable for "repetitive work with clear specifications" (like operator implementation), but core architecture design still needs to be human-led.

---

## 💭 Final Words

### Why build Kandle?

Initially, I just wanted to perform inference using onnxruntime in a JS environment. However, inference with onnxruntime requires handling a large number of intermediate tensors, which is torture in JavaScript.
Native JS methods written on the fly can only handle specific "one-off" processes. For example, it is very difficult to generically handle slicing, view transformation, or complex broadcasting operations of a high-dimensional array, and then reuse this method in the inference process of other models.

### "Why don't you use tfjs?"

Due to historical reasons, academia and cutting-edge models are mostly built using the PyTorch paradigm. In the JavaScript ecosystem, the lack of corresponding API support creates a huge cognitive switching cost when reproducing papers or porting models.

### "Why don't you use Python?"

Because I don't love writing Python; I might even say I hate it. Although Python's AI ecosystem holds a monopoly due to history, for someone accustomed to C-like languages and long-term use of strong type systems, development in Python is torture. I find it hard to get comfortable with `def` / `None` / and using "vernier calipers" (strict tools) in such a loose environment. Especially the infamous `**kwargs`—do you really know what you are writing?

Initially, I tried to export preprocessing actions (like audio Mel Spec calculation) directly as ONNX graphs. But I soon found this unfeasible; the fragmentation of the ecosystem makes model export extremely tedious. For instance, if you want to infer Whisper, there are tiny but fatal differences in preprocessing parameters across versions (e.g., turbo-v3's mel_spec n_mels is 128, while the base version is 80), meaning I would need to export specific model versions for every case. The more models I tried to infer in JS, the more obvious this "ecosystem gap" became.

### "Why don't you use transformers.js?"

I certainly tried transformers.js. It's great, works out of the box, and supports many mainstream models. But precisely because of this, it has a core problem: it is based on onnxruntime and **is a black box**. You can only adjust the Pipeline through limited parameters; you can hardly control the details of data flow. If you want to deeply customize or optimize the process, this is extremely frustrating.

### What are we missing?

After calm reflection, I discovered that the source of my pain was not the lack of a machine learning framework in JS—in fact, we have tfjs. Tfjs is powerful, but its API design philosophy stems from the previous generation of deep learning frameworks. When I want to casually write `x.view().transpose()` in JS, I find myself having to look up documentation that feels slightly alien to me. It's good, but it's not the 'standard' I'm used to.

We also don't lack inference frameworks; onnxruntime / WebLLM have done deep optimizations.

**We are missing PyTorch**, or rather, a de facto API that conforms to the Torch standard.

Naturally, I searched for existing libraries like torch-js. Although they did a lot of Binding work for ATen and c10 (PyTorch's core C++ libraries), unfortunately, most did not complete the work, and many projects stopped maintenance years ago. This was undoubtedly even more frustrating.

Is there really no way? Can we really only go back to writing Python?

### The Possibility Brought by AI

The good news is, I've been writing C++ for years. The better news is, in this era, we have a "cheat code"—AI.

I cloned the torch repository and read the ATen and c10 source code deeply. With Gemini's help, I roughly understood their design: the Dispatch system, the code generation system, the philosophy of separating storage and computation, etc. This architecture is powerful, but also very complex.

Then, I tried to replicate a simplified version in TypeScript.

### The Journey of Tearing Down and Rebuilding

This journey was a series of MVP versions being torn down and rebuilt:

* From a crude version that could only calculate `T + T`, gradually implementing type promotion and scalar calculation.
* From chaotic data type management to a clear distinction between Logical Dtype and Physical Dtype.
* From strictly operating on contiguous memory to learning how to calculate Stride, implementing memory views and dimension folding.
* Then to implementing broadcasting, advanced indexing.
* Finally completing the separation of storage and computation, backend isolation, and user-friendly API design...

Finally, I was convinced I held all the necessary pieces of tensor computation.

The rest became pure: write documentation, design prompts, and direct AI to land the code.

Development was not smooth sailing. Every time a layer of abstraction was missed, or a key design was not considered, it often meant a subsequent large-scale refactor. Of course, I also had to fight the "model hallucinations" specific to the Vibe Coding mode. Even with vibe coding, it was painful.

But as long as I thought **"If I don't do this, I have to go back to writing Python"**, I had infinite energy.

And then, there was Kandle.

### But I Must Clarify

**About Autograd**: Initially, to focus on inference scenarios and reduce engineering effort, I cut Autograd. In fact, it was hard for me to imagine large-scale model training scenarios in JS. But since the logic of backpropagation essentially shares the same dispatch system as forward propagation, and having come this far, it would be a shame not to do Autograd. With this mindset, I finally decided to patch Autograd in.

**About Maturity**: The current Kandle is still a "toy," at most a "delicate toy." Because the core is eager mode, more application scenarios are in intermediate calculation/preprocessing/post-processing, and state checking. Unless the target model cannot be exported to onnx or transformers.js has not implemented the pipeline, if directly applied to specific inference business, the performance will be far lower than specialized inference frameworks.

But eager mode also has advantages that black boxes cannot replace. **Now you can completely "dissect" the model; you can fully control every layer, every forward propagation**. I also designed hooks in nn.Module. For example, it can now be combined with DOM APIs, giving us cooler, more intuitive ways to perform model interpretability analysis. In the new version, I tried rewriting nn.Module using generator/yield; now you can "hand over" control during propagation, and you can "pause" the calculation of a certain layer at any time to debug like hitting a breakpoint in an IDE.

### Future Possibilities

I still have many ideas that haven't been implemented yet. For example, due to the thorough decoupling, I can now implement a **Remote Backend**, interacting via gRPC/WebSocket schemes. Just like calling WebGPU on the Web, the user side issues calculations and only blocks to get data at "synchronization points." This is theoretically completely feasible.

I personally believe **this design of Torch can completely become a "tensor computation protocol" standard**, not just a machine learning framework under Python, but capable of achieving much more.

Of course, with my current ability, I can only get this far. After I "evolve," I will try to go further.

---

## 📄 License

MIT License

---

<div align="center">

**🌟 If this project helps you, please give it a Star!**

**💬 If you have any thoughts on "JavaScript Version of PyTorch", welcome to share in Issues/Discussions**

Made with ❤️ by Vibe Coding

</div>
