# Kandle 已知问题与解决方案

##  WebGPU GELU 激活函数产生 NaN

### 问题描述

**影响范围**: 使用 WebGPU 后端进行推理的所有模型（Whisper、Qwen3 等）  
**严重程度**: 🔴 Critical - 导致模型推理完全失败

在 Whisper encoder 第 4 层（从 0 开始计数为 layer 3）的 MLP 模块中，GELU 激活函数输出产生 NaN 值，导致后续所有层的计算全部变成 NaN，最终推理结果无效。

### 问题现象

```
[WhisperAudioEncoder] 正在通过第 3 层 Transformer
Input: Float32Array(10) [ -0.432, 0.241, 0.109, ... ]  // 正常输入
[DEBUG] fc1 output: max=83.45, min=-39.08, nanCount=0  // fc1 输出正常，但数值较大
[DEBUG] gelu output: hasNaN=true, nanCount=13641/3072000  // GELU 产生部分 NaN

[WhisperAudioEncoder] 正在通过第 4 层 Transformer
Input: Float32Array(10) [ NaN, NaN, NaN, ... ]  // 输入已全部 NaN
```

### 根本原因分析

#### 1. 直接原因

WebGPU 后端的 GELU 实现使用 tanh 近似公式：

```wgsl
// 原实现 (有问题)
gelu(x) = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x³)))
```

当 fc1 输出的值较大时（例如 x = -15.9 或 x = 83.45）：

1. **x³ 计算产生极大值**: 当 x = -15.9 时，x³ ≈ -4020
2. **tanh 参数溢出**: `0.7978845608 * (x + 0.044715 * x³)` ≈ -156
3. **tanh 内部溢出**: 某些 GPU 驱动/硬件实现 `tanh(z)` 时使用 `(exp(z) - exp(-z)) / (exp(z) + exp(-z))`，当 z = -156 时，`exp(156)` 会溢出产生 `Infinity`，导致 `Infinity / Infinity = NaN`

#### 2. 为什么只在特定层出现？

- **层 0-2**: fc1 输出 max 约 6-8，GELU 正常
- **层 3**: fc1 输出 max=83.45，min=-39.08（数值范围突然增大）
- 这是模型权重和激活值累积的结果，某些层的权重组合会放大激活值

#### 3. 为什么只有部分元素是 NaN？

在 3,072,000 个元素中只有 13,641 个是 NaN（约 0.44%），说明只有那些输入值落在 "危险区间"（约 |x| > 10-15）的元素才会触发溢出。

### 解决方案

#### 修复代码

**文件**: `packages/backend-webgpu/src/kernels/pointwise/ops.ts`

```typescript
// 修复前
'gelu': { 
  expr: (i) => `(0.5 * ${i[0]} * (1.0 + tanh(0.7978845608 * (${i[0]} + 0.044715 * ${i[0]} * ${i[0]} * ${i[0]}))))` 
},

// 修复后 - 添加 clamp 确保 tanh 输入在安全范围内
'gelu': { 
  expr: (i) => `(0.5 * ${i[0]} * (1.0 + tanh(clamp(0.7978845608 * (${i[0]} + 0.044715 * ${i[0]} * ${i[0]} * ${i[0]}), -20.0, 20.0))))` 
},
```

#### 为什么 clamp 到 [-20, 20] 是安全的？

- `tanh(20) ≈ 0.9999999958776927` ≈ 1.0
- `tanh(-20) ≈ -0.9999999958776927` ≈ -1.0
- 超出这个范围的值对 tanh 输出几乎没有影响，但会导致数值不稳定

---

## 全面排查与预防指南

### 一、常见 NaN 产生场景

| 场景 | 触发条件 | 典型位置 | 修复方法 |
|------|----------|----------|----------|
| **exp 溢出** | 输入 > 88 (float32) | softmax, sigmoid, gelu | 减去 max 值或 clamp 输入 |
| **log 下溢** | 输入 ≤ 0 | log_softmax, cross_entropy | 添加 epsilon 或 clamp |
| **除零** | 分母 = 0 | layer_norm, rms_norm, softmax | 添加 epsilon |
| **sqrt 负数** | 输入 < 0 | layer_norm (variance) | 使用 abs 或 clamp |
| **tanh/sigmoid 溢出** | 极大输入值 | gelu, silu, mish | clamp 输入范围 |
| **大数相乘** | 权重或激活值过大 | matmul, linear | 使用混合精度或梯度裁剪 |

### 二、需要排查的激活函数

以下函数在 `packages/backend-webgpu/src/kernels/pointwise/ops.ts` 中都需要检查：

```typescript
// ⚠️ 潜在风险函数
'sigmoid': (1.0 / (1.0 + exp(-x)))     // exp(-x) 当 x 很大时可能溢出
'silu': (x / (1.0 + exp(-x)))          // 同上
'softplus': log(1.0 + exp(x))          // exp(x) 当 x > 88 时溢出
'mish': (x * tanh(log(1.0 + exp(x))))  // 多重风险

// ✅ 已修复
'gelu': 添加了 clamp
```

### 三、推荐的稳定实现

```typescript
// sigmoid - 使用 clamp 防止 exp 溢出
'sigmoid': { 
  expr: (i) => `(1.0 / (1.0 + exp(-clamp(${i[0]}, -88.0, 88.0))))` 
},

// silu - 同样需要 clamp
'silu': { 
  expr: (i) => `(${i[0]} / (1.0 + exp(-clamp(${i[0]}, -88.0, 88.0))))` 
},

// softplus - 分段实现更稳定
// 当 x > 20 时，log(1 + exp(x)) ≈ x
'softplus': { 
  expr: (i) => `select(log(1.0 + exp(${i[0]})), ${i[0]}, ${i[0]} > 20.0)` 
},

// mish - 组合多个稳定化技巧
'mish': { 
  expr: (i) => {
    const x = i[0];
    // softplus 稳定版
    const sp = `select(log(1.0 + exp(${x})), ${x}, ${x} > 20.0)`;
    // tanh 稳定版
    return `(${x} * tanh(clamp(${sp}, -20.0, 20.0)))`;
  }
},
```

### 四、Normalize 操作的稳定性检查

检查 `packages/backend-webgpu/src/kernels/normalize/` 目录：

```typescript
// softmax - 必须减去 max 值防止 exp 溢出
// 正确实现: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

// layer_norm - 除法必须加 epsilon
// 正确实现: (x - mean) / sqrt(var + eps)

// rms_norm - 同样需要 epsilon
// 正确实现: x / sqrt(mean(x²) + eps)
```

### 五、调试工具建议

#### 1. 添加 NaN 检测工具函数

```typescript
// packages/core/src/utils/debug.ts
export async function checkNaN(tensor: Tensor, name: string): Promise<boolean> {
  const data = await tensor.dataAsync() as Float32Array;
  let hasNaN = false, hasInf = false;
  let max = -Infinity, min = Infinity, nanCount = 0;
  
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v)) { hasNaN = true; nanCount++; }
    else if (!Number.isFinite(v)) hasInf = true;
    else { max = Math.max(max, v); min = Math.min(min, v); }
  }
  
  if (hasNaN || hasInf) {
    console.warn(`[NaN Check] ${name}: hasNaN=${hasNaN}, hasInf=${hasInf}, ` +
                 `nanCount=${nanCount}/${data.length}, max=${max}, min=${min}`);
  }
  return hasNaN || hasInf;
}
```

#### 2. 在关键位置添加断言

```typescript
// 在 Module.forward() 返回前检查
async forward(input: Tensor): Promise<Tensor> {
  const output = /* ... */;
  
  if (process.env.DEBUG_NAN === 'true') {
    await checkNaN(output, `${this.constructor.name}.forward`);
  }
  
  return output;
}
```

### 六、单元测试建议

为每个激活函数添加边界值测试：

```typescript
describe('GELU numerical stability', () => {
  it('should handle large positive values', async () => {
    const input = tensor([50, 100, 200]);
    const output = functional.gelu(input);
    const data = await output.dataAsync();
    expect(data.every(v => Number.isFinite(v))).toBe(true);
  });
  
  it('should handle large negative values', async () => {
    const input = tensor([-50, -100, -200]);
    const output = functional.gelu(input);
    const data = await output.dataAsync();
    expect(data.every(v => Number.isFinite(v))).toBe(true);
  });
});
```

### 七、长期预防措施

1. **代码审查检查项**: 任何包含 `exp`, `log`, `tanh`, `sqrt` 的 shader 代码必须检查输入范围
2. **CI 测试**: 添加极值输入的自动化测试
3. **运行时检测**: 开发模式下自动检测 NaN 并报警
4. **文档**: 在每个激活函数实现旁注明数值范围限制

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `packages/backend-webgpu/src/kernels/pointwise/ops.ts` | GELU 添加 clamp |
| `packages/model-utils/src/whisper/block.ts` | 移除调试代码（完成后） |

---

## 参考资料

- [WGSL 规范 - 内置函数](https://www.w3.org/TR/WGSL/#built-in-functions)
- [PyTorch 数值稳定性指南](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [GELU 论文](https://arxiv.org/abs/1606.08415)
