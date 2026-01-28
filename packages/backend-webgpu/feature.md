# WebGPU 后端特性文档

本文档详细说明了 kandle WebGPU 后端的特殊处理机制、已知限制以及用户需要注意的事项。

---

## 📋 目录

1. [核心特性](#核心特性)
2. [类型转换系统](#类型转换系统)
3. [设备功能检测](#设备功能检测)
4. [复数类型支持](#复数类型支持)
5. [Shape 语义](#shape-语义)
6. [性能注意事项](#性能注意事项)
7. [已知限制](#已知限制)

---

## 🎯 核心特性

### 异步数据访问
- WebGPU buffer 只能异步读取
- 必须使用 `tensor.dataAsync()` 而非 `tensor.data`
- GPU 计算完成后需要等待结果复制回 CPU

### Shader 编译
- 所有运算需要预编译 WGSL shader
- Pipeline 被缓存以提升后续运算性能
- 缓存 Key 包含：`opName-inputDtypes-commonDtype-rank`

---

## 🔄 类型转换系统

由于 WebGPU/WGSL 原生支持的类型有限，我们通过 **DTypeResolver** 架构在 **客户端侧** 进行类型转换，使用户可以透明地使用所有 kandle 支持的类型。

### DTypeResolver 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     类型层次架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户视角 (Logical DType)                                   │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│   │ float16 │  │ float32 │  │ int8    │  ...               │
│   └────┬────┘  └────┬────┘  └────┬────┘                    │
│        │            │            │                          │
│        ▼            ▼            ▼                          │
│   ┌────────────────────────────────────────┐               │
│   │         DType Resolver (初始化时)       │               │
│   │  - 检测设备能力 (shader-f16)            │               │
│   │  - 确定物理存储策略                      │               │
│   │  - 生成转换函数                          │               │
│   └────────────────────────────────────────┘               │
│        │            │            │                          │
│        ▼            ▼            ▼                          │
│   GPU 视角 (Physical DType)                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│   │  f16    │  │   f32   │  │   i32   │  ...               │
│   │ 或 f32  │  │         │  │         │                    │
│   └─────────┘  └─────────┘  └─────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计

```typescript
// PhysicalStorageDescriptor - 每种 dtype 的物理存储描述
interface PhysicalStorageDescriptor {
    logicalDType: DType;           // 逻辑类型
    wgslStorageType: WgslDType;    // GPU 存储类型 (f16, f32, i32, u32, vec2<f32>)
    wgslComputeType: WgslDType;    // GPU 计算类型
    gpuBytesPerElement: number;    // 每元素字节数
    jsTypedArrayCtor: TypedArrayConstructor;  // JS 端返回的数组类型
    uploadConverter: DataConverter;   // 上传转换函数
    downloadConverter: DataConverter; // 下载转换函数
}

// 使用方式
const resolver = getGlobalDTypeResolver();
const desc = resolver.getDescriptor('float16');
// desc.wgslStorageType -> 'f16' (如果设备支持) 或 'f32' (fallback)
```

### 类型转换矩阵

| 逻辑类型 | JS TypedArray | WebGPU 物理存储 | 转换策略 |
|----------|---------------|-----------------|----------|
| `float32` | `Float32Array` | `array<f32>` | 直接使用 ✅ |
| `float64` | `Float64Array` | `array<f32>` | **降级**：f64 → f32 (精度损失) |
| `float16` | `Float32Array` | `array<f16>` 或 `array<f32>` | 设备相关，详见下文 |
| `int32` | `Int32Array` | `array<i32>` | 直接使用 ✅ |
| `uint32` | `Uint32Array` | `array<u32>` | 直接使用 ✅ |
| `int64` | `BigInt64Array` | `array<i32>` | **降级**：i64 → i32 (溢出风险) |
| `uint64` | `BigUint64Array` | `array<u32>` | **降级**：u64 → u32 (溢出风险) |
| `int16` | `Int16Array` | `array<i32>` | **扩展**：i16 → i32 |
| `uint16` | `Uint16Array` | `array<u32>` | **扩展**：u16 → u32 |
| `int8` | `Int8Array` | `array<i32>` | **扩展**：每个 i8 存为 1 个 i32 |
| `uint8` | `Uint8Array` | `array<u32>` | **扩展**：每个 u8 存为 1 个 u32 |
| `bool` | `Uint8Array` | `array<u32>` | **扩展**：每个 bool 存为 1 个 u32 |
| `complex64` | `Float32Array` | `array<vec2<f32>>` | 直接使用 ✅ ([real, imag] pairs) |
| `complex128` | `Float64Array` | `array<vec2<f32>>` | **降级**：每对 f64 → f32 |

### 上传/下载转换

```typescript
// 用户代码 (创建 tensor)
const tensor = new Tensor(new Int8Array([1, 2, 3, 4]), { dtype: 'int8' });

// 内部处理 (通过 DTypeResolver):
// 1. resolver.getDescriptor('int8').uploadConverter() 将 int8 扩展为 i32
// 2. 上传到 GPU 的是 Int32Array [1, 2, 3, 4]

// 用户代码 (读取数据)
const data = await tensor.dataAsync(); // 返回 Int8Array

// 内部处理:
// 1. 从 GPU 读取 Int32Array
// 2. resolver.getDescriptor('int8').downloadConverter() 压缩回 Int8Array
```

### ⚠️ 用户注意事项

1. **float64/int64/uint64 精度损失**
   - 这些类型会降级为 32-bit 进行计算
   - 结果会自动转换回 64-bit，但精度已丢失
   - 如果需要高精度计算，考虑使用 JS 后端

2. **int8/uint8/int16/uint16 存储开销**
   - 为了避免 Shader 中的 Data Race，小类型会扩展为 32-bit
   - 内部存储空间 = `numel * 4` bytes (而非原始大小)

---

## 🔍 设备功能检测

### DTypeResolver 初始化

设备能力检测在 `WebGPUDeviceManager.init()` 时完成，然后构建全局 DTypeResolver：

```typescript
// device.ts 中
static async init(): Promise<void> {
    // ... 设备初始化 ...
    
    // 检测 shader-f16 支持
    this._supportsF16 = adapter.features.has('shader-f16');
    
    // 构建 DTypeResolver (根据设备能力确定物理存储策略)
    initGlobalDTypeResolver(this._supportsF16);
}
```

### Float16 (shader-f16) 支持

通过 DTypeResolver 查询设备能力：

```typescript
const resolver = getGlobalDTypeResolver();
resolver.supportsNativeF16;            // boolean
resolver.float16StoragePrecision;      // 'f16' | 'f32'
```

#### 支持 f16 时
- `float16` 直接使用 `f16` 类型存储
- Shader 添加 `enable f16;` 指令
- `uploadConverter`: Float32 → f16 bits (Uint16)
- `downloadConverter`: f16 bits → Float32

#### 不支持 f16 时 (Fallback)
- `float16` 数据存储为 `f32`
- Shader 使用 `f32` 进行计算
- 无需额外转换（用户提供 Float32Array，直接存储）
- 用户无需关心，类型语义保持一致

### 检测的功能列表

```typescript
// 初始化时输出
console.log(`[WebGPU] Adapter features:`, [...adapter.features]);
console.log(`[WebGPU] shader-f16 support: ${resolver.supportsNativeF16}`);
```

---

## 🔢 复数类型支持

### 存储格式

- `complex64`: `vec2<f32>` = 2 × float32 = 8 bytes/element
- `complex128`: `vec2<f32>` = 2 × float32 = 8 bytes/element (降级后)

### 数据布局

```typescript
// Float32Array: [real0, imag0, real1, imag1, ...]
const data = new Float32Array([1, 2, 3, 4]); // (1+2i), (3+4i)
const tensor = new Tensor(data, { dtype: 'complex64', shape: [2] });
```

### 复数运算实现

| 运算 | 公式 | 实现方式 |
|------|------|----------|
| `+` | `(a+bi) + (c+di) = (a+c) + (b+d)i` | vec2 直接加法 |
| `-` | `(a+bi) - (c+di) = (a-c) + (b-d)i` | vec2 直接减法 |
| `*` | `(a+bi) × (c+di) = (ac-bd) + (ad+bc)i` | 特殊 WGSL 公式 |
| `/` | `(a+bi) / (c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)` | 特殊 WGSL 公式 |

### 实数 + 复数 类型提升

```typescript
const real = new Tensor([1, 2], { dtype: 'float32' });
const complex = new Tensor([3, 4, 5, 6], { dtype: 'complex64', shape: [2] });

const result = real.add(complex);
// 结果类型: complex64
// 实数被提升为虚部为 0 的复数
```

---

## 📐 Shape 语义

### 标量类型

`shape` 直接等于元素数量：

```typescript
new Tensor([1, 2, 3, 4], { dtype: 'float32' })
// shape: [4], data.length: 4
```

### 复数类型

`shape` 表示**复数元素数量**，不是底层 float 数量：

```typescript
new Tensor(new Float32Array([1, 2, 3, 4]), { dtype: 'complex64', shape: [2] })
// shape: [2] (2 个复数)
// data.length: 4 (每个复数 2 个 float)
```

### 验证规则

```typescript
const isComplex = dtype === 'complex64' || dtype === 'complex128';
const expectedDataLength = isComplex ? numel * 2 : numel;
```

---

## ⚡ 性能注意事项

### Pipeline 缓存

```typescript
// 缓存 Key 格式
`binary.${opName}-${inputDtypes.join('-')}-${commonDtype}-${isContiguous ? 'fast' : `general-r${rank}`}`

// 例如：
// "binary.add-float32-float32-float32-fast"
// "binary.mul-int8-int32-int32-general-r3"
```

- 相同类型+shape+运算 复用已编译的 Pipeline
- 不同类型组合会产生新的 Pipeline
- 首次运算有编译延迟，后续运算更快

### 内存对齐

- 所有 buffer 自动对齐到 4 字节边界
- int8/uint8 打包后实际占用空间可能更大
- 考虑批量处理以摊薄开销

### 异步计算

```typescript
// ❌ 错误：同步访问 GPU 数据
const data = tensor.data;  // 可能失败或返回空

// ✅ 正确：异步访问
const data = await tensor.dataAsync();
```

---

## ⚠️ 已知限制

### 1. 精度限制

| 类型 | 限制 |
|------|------|
| `float64` | 降级为 float32，16位有效数字 → ~7位 |
| `int64/uint64` | 降级为 32-bit，超出范围会溢出 |
| `complex128` | 降级为 complex64 精度 |

### 2. 不支持的操作

- 目前仅支持二元算术运算 (`add`, `sub`, `mul`, `div`)
- 归约运算 (`sum`, `mean`) 和一元运算 (`sin`, `cos`) 待实现

### 3. Broadcasting 限制

- 支持标准 NumPy-style broadcasting
- Fast path 仅适用于连续内存布局
- Non-contiguous tensor 使用 general path（稍慢）

### 4. 设备兼容性

- 需要 WebGPU 支持 (Chrome 113+, Edge 113+, Firefox Nightly)
- `shader-f16` 需要硬件支持
- 移动设备支持有限

### 5. Buffer Aliasing (In-place 操作限制)

**问题**：WebGPU 不允许同一个 GPUBuffer 同时绑定为 `read-only-storage` (输入) 和 `storage` (输出)。

当执行原地操作如 `a.add(b, a)` 时：
- `a` 既是输入 (binding 1) 又是输出 (binding 3)
- WebGPU validation 会阻止这种绑定，导致命令静默失败

**我们的解决方案**：

```typescript
// executor.ts 中检测并处理 buffer aliasing
if (bufferA === outputBuffer) {
    // 创建临时 buffer 并复制输入数据
    tempBufferA = device.createBuffer({ size: bufferA.size, ... });
    copyEncoder.copyBufferToBuffer(bufferA, 0, tempBufferA, 0, bufferA.size);
    bufferA = tempBufferA;
}
```

**处理流程**：
1. 检测 input buffer 是否与 output buffer 相同
2. 如果相同，创建临时 buffer 并复制输入数据
3. 使用临时 buffer 作为输入执行计算
4. 结果写入原始 buffer (out)
5. 临时 buffer 由 JS GC 自动回收

**用户注意事项**：
- ✅ `a.add(b, a)` 可以正常工作（原地操作）
- ✅ `a.add(b, b)` 可以正常工作
- ✅ `a.add(b, c)` 可以正常工作（c 是独立 tensor）
- ⚠️ 原地操作有额外的 buffer 复制开销
- 🚫 不支持部分重叠的 view（如 slice）作为 out 参数

---

## 🛠️ 我们做的"脏活累活"

### 1. DTypeResolver 架构
- 设计并实现了 **逻辑类型 ↔ 物理类型** 分离机制
- 在设备初始化时一次性确定所有类型的处理策略，消除运行时分支
- 统一的 `PhysicalStorageDescriptor` 接口，支持 O(1) 查表

### 2. 类型转换层
- 实现了 `float16ToFloat32` / `float32ToFloat16` 位操作转换
- 处理 IEEE 754 半精度浮点数的符号、指数、尾数
- 支持非规格化数、无穷大、NaN

### 3. 小类型扩展
- int8/uint8/int16/uint16 扩展为 32-bit 存储 (避免 Data Race)
- `uploadConverter` / `downloadConverter` 自动处理类型转换
- 确保负数（如 int8 的 -128）正确处理

### 4. 复数运算
- 手写复数乘除法公式
- vec2<f32> 存储格式适配
- 实数到复数的隐式提升

### 5. 设备能力感知类型解析
- DTypeResolver 根据 `shader-f16` 动态选择 float16 策略
- Shader 代码通过 `resolver.supportsNativeF16` 自动添加 `enable f16;`
- Pipeline 缓存 key 包含类型信息防止冲突

### 6. 类型提升逻辑
- 完整实现 PyTorch 风格的类型提升规则
- 整数除法自动转浮点
- 跨位宽整数运算自动提升

### 7. Buffer Aliasing 处理
- 检测 in-place 操作导致的 input/output buffer 冲突
- 自动创建临时 buffer 并复制数据以绕过 WebGPU 限制
- 保证 `a.add(b, a)` 等原地操作正确执行

---

## 📞 反馈与贡献

如果遇到问题或有改进建议，欢迎提交 Issue 或 PR！
