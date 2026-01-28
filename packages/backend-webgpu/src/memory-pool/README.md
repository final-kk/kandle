# GPU Memory Pool (已弃用)

> **状态**：已弃用 (Deprecated)  
> **保留原因**：作为设计参考和历史记录

## 为什么不启用这个设计？

### 1. Chrome Dawn 已实现底层子分配

通过研究 TensorFlow.js 的 WebGPU 后端实现，我们发现：

- **TFJS 没有在 JS 层实现 Slab Allocator**：它的 `BufferManager` 只是 `device.createBuffer` 的薄包装
- **Chrome 的 Dawn 引擎已经在底层实现了子分配**：小 Buffer（<4MB）在大 Heap 中子分配
- **`buffer.destroy()` 不会立即释放显存**：Dawn 将其标记为"空闲"供后续复用

这意味着我们在 JS 层实现 Arena-based Memory Pool 是 **重复劳动**。

### 2. 显存"驻留"是正常行为

观察到的现象：
- 第一次运行后显存停留在高位（如 1.5GB）
- 后续运行不再增长

这不是内存泄漏，而是 Dawn 的缓存策略。真正的释放只在以下情况发生：
- WebGPU context 销毁
- 页面刷新
- 使用 Chrome flag `--enable-dawn-features=disable_resource_suballocation`（会影响性能）

### 3. 简化设计足够满足需求

当前采用的简化方案 (`base/storage.ts`)：

| 特性 | 简化方案 | 本目录 Arena 方案 |
|:----|:---------|:-----------------|
| 内存分配 | 直接 `createBuffer` | Arena 子分配 |
| 引用计数 | ✅ `incRef()`/`decRef()` | ✅ AllocationStore |
| GC 兜底 | ✅ FinalizationRegistry | ✅ 同上 |
| Uniform Pool | ✅ UniformBufferPool | ✅ 更复杂版本 |
| 复杂度 | ~200 行 | ~1000+ 行 |
| 维护成本 | 低 | 高 |

### 4. TFJS 的启示

> TFJS 的做法：疯狂调用 `createBuffer` → 用完 → `dispose` (destroy)  
> Chrome 底层：在同一个大 Heap 里反复擦写

这种"简单粗暴"的方法在 TFJS 这样的生产级推理库中被验证有效。

## 本目录包含的组件（供参考）

| 文件 | 功能 | 复杂度 |
|:----|:----|:------|
| `arena.ts` | Arena Buffer 管理 | 高 |
| `arena-manager.ts` | 多 Arena 池管理 | 高 |
| `allocation-store.ts` | 分配元数据 + RefCount | 中 |
| `free-lists.ts` | Segregated Free Lists | 中 |
| `compaction.ts` | Arena 碎片整理 | 高 |
| `fence-tracker.ts` | GPU 同步追踪 | 中 |
| `staging-pool.ts` | Staging Buffer 池 | 中 |
| `pooled-storage.ts` | 池化 Storage 实现 | 中 |

## 何时考虑启用？

如果出现以下情况，可能需要重新评估：

1. **Dawn 行为变化**：未来 Chrome 版本改变了子分配策略
2. **跨平台一致性**：需要在非 Chrome 环境（Firefox/Safari）上获得一致的内存行为
3. **极端内存压力**：需要更精细的内存控制

## 相关设计文档

- `design/gpu-memory-pool-v8-final.md` - 完整设计规范
- `design/gpu-memory-refcount-arc.md` - 引用计数方案

---

*最后更新：2026-01-04*
