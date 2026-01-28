# NN-Kit I/O 模块

> 版本: 1.0  
> 日期: 2026-01-01

工业级的数据加载模块，支持 Safetensors 和 NPY 格式。

## 核心特性

- **分片支持** - 支持 `model.safetensors.index.json` 分片模型
- **零拷贝设计** - Descriptor 不持有数据，按需加载
- **平台抽象** - ByteSource 统一 Web/Node.js 接口
- **取消机制** - AbortSignal 支持
- **并发控制** - batchSize 避免浏览器限制

## 快速上手

### 1. 加载单个 Safetensor 文件

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

### 2. 加载分片模型

```typescript
import { io } from '@kandle/core';

// 自动检测分片模型
const group = await io.loadSafetensor('./model.safetensors.index.json');

console.log(`分片数: ${group.files.size}`);
console.log(`层数: ${group.layers.size}`);

// 无需关心物理分片，统一访问
for (const [name, layer] of group.layers) {
    console.log(`${name} @ ${layer.file.path}`);
}
```

### 3. Module 权重加载

```typescript
import { io, nn } from '@kandle/core';

// 定义模型
class MyModel extends nn.Module {
    linear: nn.Linear;
    
    constructor() {
        super();
        this.linear = new nn.Linear(768, 768);
        this._registerChildren();
    }
    
    async forward(x: Tensor) {
        return this.linear.call(x);
    }
}

// 加载权重
const model = new MyModel();
const group = await io.loadSafetensor('./model.safetensors');

const result = await model.loadFromSafetensor(group, {
    strict: true,
    device: 'webgpu',
    batchSize: 4,  // 并发控制
});

console.log(`加载: ${result.loadedKeys.length}`);
console.log(`缺失: ${result.missingKeys}`);
console.log(`多余: ${result.unexpectedKeys}`);
```

### 4. 键名映射

```typescript
const result = await model.loadFromSafetensor(group, {
    // 自定义键名映射
    keyMapper: (key) => key.replace('model.', ''),
});
```

### 5. 加载 NPY 文件

```typescript
import { io } from '@kandle/core';

const desc = await io.loadNpy('./weights.npy');
console.log(`Shape: [${desc.shape}]`);
console.log(`DType: ${desc.dtype}`);

const tensor = await io.tensorFromNpy(desc, { device: 'webgpu' });
```

### 6. 取消加载

```typescript
const controller = new AbortController();

// 用户点击取消
cancelButton.onclick = () => controller.abort();

try {
    const group = await io.loadSafetensor('./large_model.safetensors', controller.signal);
} catch (e) {
    if (e.name === 'AbortError') {
        console.log('加载已取消');
    }
}
```

## API 参考

### ByteSource

平台抽象层，用于读取原始字节。

```typescript
interface ByteSource {
    read(offset: number, length: number, signal?: AbortSignal): Promise<ArrayBuffer>;
    size(signal?: AbortSignal): Promise<number>;
    close(): void;
}

interface ResolvableByteSource extends ByteSource {
    resolve(relativePath: string): ByteSource;
}
```

**实现类**:
- `WebByteSource` - fetch + Range request
- `ArrayBufferByteSource` - 内存 ArrayBuffer
- `FileByteSource` - 浏览器 File API

### SafetensorGroup

Safetensor 文件或分片组的描述符。

```typescript
interface SafetensorGroup {
    readonly sharded: boolean;
    readonly metadata: Record<string, string>;
    readonly totalSize?: number;
    readonly files: ReadonlyMap<string, SafetensorFile>;
    readonly layers: ReadonlyMap<string, SafetensorLayer>;
    
    getLayer(name: string): SafetensorLayer | undefined;
    hasLayer(name: string): boolean;
    dumpWeightMap(): void;
    close(): void;
}
```

### SafetensorLayer

单个 tensor 的描述符。

```typescript
interface SafetensorLayer {
    readonly name: string;
    readonly dtype: DType;
    readonly shape: readonly number[];
    readonly originalDtype: SafetensorsDType;
    readonly file: SafetensorFile;
    readonly dataOffsets: readonly [number, number];
    readonly byteSize: number;
    readonly numel: number;
}
```

### 加载函数

```typescript
// 加载 Safetensor
function loadSafetensor(
    source: string | ArrayBuffer | File,
    signal?: AbortSignal
): Promise<SafetensorGroup>;

// 从 Layer 创建 Tensor
function tensorFromSafetensorLayer(
    layer: SafetensorLayer,
    options?: TensorLoadOptions,
    signal?: AbortSignal
): Promise<Tensor>;

// 加载 NPY
function loadNpy(
    source: string | ArrayBuffer | File,
    signal?: AbortSignal
): Promise<NpyDescriptor>;

// 从 NPY 创建 Tensor
function tensorFromNpy(
    descriptor: NpyDescriptor,
    options?: TensorLoadOptions,
    signal?: AbortSignal
): Promise<Tensor>;
```

### Module.loadFromSafetensor

```typescript
interface LoadFromSafetensorOptions {
    strict?: boolean;           // 严格匹配键 (default: true)
    device?: DeviceNameEnum;    // 目标设备
    dtype?: DType;              // 目标 dtype
    keyMapper?: (key: string) => string;  // 键名映射
    signal?: AbortSignal;       // 取消信号
    batchSize?: number;         // 并发批次 (default: 4)
}

class Module {
    async loadFromSafetensor(
        group: SafetensorGroup,
        options?: LoadFromSafetensorOptions
    ): Promise<LoadStateDictResult>;
}
```

## DType 映射

| Safetensors | NN-Kit | 说明 |
|-------------|--------|------|
| F64 | float64 | - |
| F32 | float32 | - |
| F16 | float16 | - |
| BF16 | float32 | 自动转换 |
| I64 | int64 | BigInt64Array |
| I32 | int32 | - |
| I16 | int16 | - |
| I8 | int8 | - |
| U64 | uint64 | BigUint64Array |
| U32 | uint32 | - |
| U16 | uint16 | - |
| U8 | uint8 | - |
| BOOL | bool | Uint8Array |

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 0: ByteSource                          │
│   WebByteSource      ArrayBufferByteSource      FileByteSource     │
└─────────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 1: Format Parser                       │
│               SafetensorsParser           NpyParser                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 2: User API                            │
│   tensorFromSafetensorLayer()      Module.loadFromSafetensor()     │
└─────────────────────────────────────────────────────────────────────┘
```

## 注意事项

1. **BF16 转换**: BF16 数据会在读取时自动转换为 F32
2. **int64/uint64**: 使用 BigInt64Array/BigUint64Array 保持精度
3. **内存对齐**: 使用 slice() 保证 TypedArray 对齐
4. **浏览器并发**: 同域最多 6 个并发请求，使用 batchSize 控制
5. **Range 请求**: 如服务器不支持 Range，自动 fallback 到全量下载
