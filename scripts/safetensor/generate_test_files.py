#!/usr/bin/env python3
"""
Safetensors 测试文件生成器

生成各种 corner case 的 safetensors 文件用于测试 NN-Kit IO 模块

运行方式:
    conda activate mnist
    cd scripts/safetensor
    python generate_test_files.py

输出目录结构:
    test_data/
    ├── valid/                          # 有效文件
    │   ├── single/                     # 单文件测试
    │   │   ├── basic.safetensors       # 基础类型
    │   │   ├── dtypes.safetensors      # 各种 dtype
    │   │   ├── shapes.safetensors      # 各种形状
    │   │   ├── names.safetensors       # 特殊命名
    │   │   ├── metadata.safetensors    # 丰富 metadata
    │   │   └── large.safetensors       # 较大文件
    │   │
    │   └── sharded/                    # 分片测试
    │       ├── small/                  # 3 分片
    │       └── complex/                # 5 分片，各种类型混合
    │
    └── invalid/                        # 无效文件 (用于测试错误处理)
        ├── empty.safetensors           # 空文件
        ├── truncated_header.safetensors # header 截断
        ├── bad_magic.safetensors       # 错误的 magic bytes
        ├── bad_json.safetensors        # JSON 解析错误
        ├── bad_offset.safetensors      # 错误的 offset
        └── incomplete_data.safetensors # 数据不完整
"""

import torch
from safetensors.torch import save_file
import json
import os
import shutil
import struct

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "../../packages/nn-kit/public/test_data/safetensor"

# ============================================================================
# 工具函数
# ============================================================================

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def clean_and_create(path: str):
    """清理并创建目录"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def print_section(title: str):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ============================================================================
# 1. 单文件测试 - 基础类型
# ============================================================================

def generate_single_basic(output_dir: str):
    """基础单文件，简单 float32 tensor"""
    print("📦 Generating: basic.safetensors")
    
    tensors = {
        "weight": torch.randn(4, 4, dtype=torch.float32),
        "bias": torch.randn(4, dtype=torch.float32),
    }
    
    metadata = {
        "format": "pt",
        "description": "Basic test file with float32 tensors"
    }
    
    save_file(tensors, os.path.join(output_dir, "basic.safetensors"), metadata=metadata)
    
    # 保存期望值供 JS 验证
    expected = {
        "weight": tensors["weight"].tolist(),
        "bias": tensors["bias"].tolist(),
    }
    with open(os.path.join(output_dir, "basic.expected.json"), "w") as f:
        json.dump(expected, f)

# ============================================================================
# 2. 单文件测试 - 各种 DType
# ============================================================================

def generate_single_dtypes(output_dir: str):
    """测试所有支持的 dtype"""
    print("📦 Generating: dtypes.safetensors")
    
    tensors = {
        # 浮点类型
        "float64": torch.randn(2, 3, dtype=torch.float64),
        "float32": torch.randn(2, 3, dtype=torch.float32),
        "float16": torch.randn(2, 3, dtype=torch.float16),
        "bfloat16": torch.randn(2, 3, dtype=torch.bfloat16),
        
        # 整数类型 (有符号)
        "int64": torch.randint(-100, 100, (2, 3), dtype=torch.int64),
        "int32": torch.randint(-100, 100, (2, 3), dtype=torch.int32),
        "int16": torch.randint(-100, 100, (2, 3), dtype=torch.int16),
        "int8": torch.randint(-100, 100, (2, 3), dtype=torch.int8),
        
        # 整数类型 (无符号)
        "uint8": torch.randint(0, 255, (2, 3), dtype=torch.uint8),
        
        # 布尔类型
        "bool": torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool),
    }
    
    metadata = {
        "description": "All supported dtypes"
    }
    
    save_file(tensors, os.path.join(output_dir, "dtypes.safetensors"), metadata=metadata)
    
    # 保存期望值
    expected = {}
    for name, tensor in tensors.items():
        if tensor.dtype == torch.bfloat16:
            # BF16 转 F32 后保存
            expected[name] = tensor.float().tolist()
        elif tensor.dtype == torch.bool:
            expected[name] = tensor.int().tolist()  # bool 转 int
        else:
            expected[name] = tensor.tolist()
    
    with open(os.path.join(output_dir, "dtypes.expected.json"), "w") as f:
        json.dump(expected, f)

# ============================================================================
# 3. 单文件测试 - 各种形状
# ============================================================================

def generate_single_shapes(output_dir: str):
    """测试各种奇怪的形状"""
    print("📦 Generating: shapes.safetensors")
    
    tensors = {
        # 标量
        "scalar": torch.tensor(3.14159),
        
        # 1D
        "1d_small": torch.randn(5),
        "1d_large": torch.randn(1000),
        
        # 2D
        "2d_square": torch.randn(8, 8),
        "2d_rect": torch.randn(3, 7),
        "2d_single_row": torch.randn(1, 10),
        "2d_single_col": torch.randn(10, 1),
        
        # 3D
        "3d_cube": torch.randn(4, 4, 4),
        "3d_odd": torch.randn(3, 5, 7),
        
        # 4D (常见于 conv)
        "4d_nchw": torch.randn(2, 3, 4, 5),
        
        # 5D
        "5d": torch.randn(2, 2, 2, 2, 2),
        
        # 空 tensor (size 0)
        "empty_1d": torch.randn(0),
        "empty_2d": torch.randn(0, 10),
        "empty_3d": torch.randn(5, 0, 3),
        
        # 单元素
        "single_element_1d": torch.randn(1),
        "single_element_2d": torch.randn(1, 1),
        "single_element_3d": torch.randn(1, 1, 1),
        
        # 素数维度 (测试非 2^N 对齐)
        "prime_dims": torch.randn(7, 11, 13),
    }
    
    metadata = {
        "description": "Various tensor shapes including edge cases"
    }
    
    save_file(tensors, os.path.join(output_dir, "shapes.safetensors"), metadata=metadata)
    
    # 保存形状信息
    shapes = {name: list(t.shape) for name, t in tensors.items()}
    with open(os.path.join(output_dir, "shapes.expected.json"), "w") as f:
        json.dump(shapes, f)

# ============================================================================
# 4. 单文件测试 - 特殊命名
# ============================================================================

def generate_single_names(output_dir: str):
    """测试特殊的 tensor 命名"""
    print("📦 Generating: names.safetensors")
    
    tensors = {
        # 正常命名
        "model.layers.0.weight": torch.randn(2, 2),
        "model.layers.0.bias": torch.randn(2),
        "model.layers.1.weight": torch.randn(2, 2),
        
        # 深层嵌套
        "a.b.c.d.e.f.g.h.i.j": torch.randn(2, 2),
        
        # 数字索引
        "layers.0.sublayers.1.params.2": torch.randn(2, 2),
        
        # 下划线命名
        "self_attn.q_proj.weight": torch.randn(2, 2),
        "feed_forward.gate_proj": torch.randn(2, 2),
        
        # Unicode (中文、emoji)
        "测试.权重": torch.randn(2, 2),
        "emoji.🔥.tensor": torch.randn(2, 2),
        
        # 特殊字符 (safetensors 支持这些)
        "with-dash": torch.randn(2, 2),
        "with_underscore": torch.randn(2, 2),
        
        # 空格和其他字符
        "with spaces": torch.randn(2, 2),
        "with/slashes/path": torch.randn(2, 2),
        "with:colon": torch.randn(2, 2),
        
        # 短命名
        "a": torch.randn(2, 2),
        "x": torch.randn(2, 2),
        
        # 长命名
        "very_long_name_" * 10 + "end": torch.randn(2, 2),
    }
    
    metadata = {
        "description": "Special tensor names including unicode and special characters"
    }
    
    save_file(tensors, os.path.join(output_dir, "names.safetensors"), metadata=metadata)
    
    # 保存所有键名
    with open(os.path.join(output_dir, "names.expected.json"), "w") as f:
        json.dump(list(tensors.keys()), f, ensure_ascii=False)

# ============================================================================
# 5. 单文件测试 - 丰富 Metadata
# ============================================================================

def generate_single_metadata(output_dir: str):
    """测试各种 metadata"""
    print("📦 Generating: metadata.safetensors")
    
    tensors = {
        "weight": torch.randn(4, 4),
    }
    
    metadata = {
        # 基础信息
        "format": "pt",
        "version": "1.0.0",
        
        # 模型信息
        "model_name": "test-model",
        "model_type": "transformer",
        "architecture": "decoder-only",
        
        # 数值信息
        "total_params": "1000000",  # metadata 值必须是字符串
        "hidden_size": "512",
        
        # Unicode
        "description": "这是一个测试模型 🚀",
        "author": "テスト作者",
        
        # 特殊字符
        "special": "line1\nline2\ttab",
        
        # 长值
        "long_value": "x" * 1000,
        
        # 空值
        "empty": "",
        
        # JSON-like (但作为字符串)
        "config": '{"hidden_size": 512, "num_layers": 12}',
    }
    
    save_file(tensors, os.path.join(output_dir, "metadata.safetensors"), metadata=metadata)
    
    # 保存期望的 metadata
    with open(os.path.join(output_dir, "metadata.expected.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

# ============================================================================
# 6. 单文件测试 - 较大文件
# ============================================================================

def generate_single_large(output_dir: str):
    """生成稍大的文件 (约 10MB)"""
    print("📦 Generating: large.safetensors")
    
    # 创建约 10MB 的数据
    # 10MB / 4 bytes = 2.5M floats ≈ 1600x1600 float32
    tensors = {
        "large_weight": torch.randn(1600, 1600, dtype=torch.float32),
        "another_large": torch.randn(512, 2048, dtype=torch.float32),
    }
    
    metadata = {
        "description": "Large file (~10MB) for performance testing"
    }
    
    save_file(tensors, os.path.join(output_dir, "large.safetensors"), metadata=metadata)
    
    # 只保存形状，不保存实际值
    shapes = {name: list(t.shape) for name, t in tensors.items()}
    with open(os.path.join(output_dir, "large.expected.json"), "w") as f:
        json.dump(shapes, f)

# ============================================================================
# 7. 分片测试 - 简单 3 分片
# ============================================================================

def generate_sharded_small(output_dir: str):
    """简单的 3 分片模型"""
    print("📦 Generating: sharded/small/")
    
    ensure_dir(output_dir)
    
    # 定义 tensors 和它们的分片分配
    shard_contents = {
        "model-00001-of-00003.safetensors": {
            "embed.weight": torch.randn(100, 64, dtype=torch.float32),
        },
        "model-00002-of-00003.safetensors": {
            "layers.0.weight": torch.randn(64, 64, dtype=torch.float32),
            "layers.0.bias": torch.randn(64, dtype=torch.float32),
            "layers.1.weight": torch.randn(64, 64, dtype=torch.float32),
            "layers.1.bias": torch.randn(64, dtype=torch.float32),
        },
        "model-00003-of-00003.safetensors": {
            "head.weight": torch.randn(64, 10, dtype=torch.float32),
            "head.bias": torch.randn(10, dtype=torch.float32),
        },
    }
    
    # 构建 weight_map
    weight_map = {}
    for filename, tensors in shard_contents.items():
        for name in tensors.keys():
            weight_map[name] = filename
    
    # 写入各分片
    for filename, tensors in shard_contents.items():
        filepath = os.path.join(output_dir, filename)
        metadata = {"shard": filename}
        save_file(tensors, filepath, metadata=metadata)
        print(f"  📄 {filename}: {len(tensors)} tensors")
    
    # 写入 index.json
    index = {
        "metadata": {
            "total_size": 0,  # 简化，不计算
        },
        "weight_map": weight_map
    }
    
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"  📜 model.safetensors.index.json: {len(weight_map)} entries")

# ============================================================================
# 8. 分片测试 - 复杂 5 分片
# ============================================================================

def generate_sharded_complex(output_dir: str):
    """复杂的 5 分片模型，各种 dtype 混合"""
    print("📦 Generating: sharded/complex/")
    
    ensure_dir(output_dir)
    
    NUM_SHARDS = 5
    
    # 定义所有 tensors
    all_tensors = {
        # 分片 1: Embedding
        "model.embed_tokens.weight": torch.randn(1000, 128, dtype=torch.float16),
        
        # 分片 2: Layer 0
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128, dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 512, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(128, 512, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 128, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.randn(128, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(128, dtype=torch.float32),
        
        # 分片 3: Layer 1
        "model.layers.1.self_attn.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(128, 512, dtype=torch.bfloat16),
        "model.layers.1.mlp.up_proj.weight": torch.randn(128, 512, dtype=torch.bfloat16),
        "model.layers.1.mlp.down_proj.weight": torch.randn(512, 128, dtype=torch.bfloat16),
        "model.layers.1.input_layernorm.weight": torch.randn(128, dtype=torch.float32),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(128, dtype=torch.float32),
        
        # 分片 4: Norm + Head
        "model.norm.weight": torch.randn(128, dtype=torch.float32),
        "lm_head.weight": torch.randn(1000, 128, dtype=torch.float16),
        
        # 分片 5: 杂项
        "model.vocab_ids": torch.randint(0, 1000, (100,), dtype=torch.int64),
        "model.attention_mask": torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.bool),
        "special.测试": torch.randn(2, 2, dtype=torch.float32),
        "special.🚀": torch.randn(2, 2, dtype=torch.float32),
    }
    
    # 分配策略
    shard_assignment = {
        "model-00001-of-00005.safetensors": [
            "model.embed_tokens.weight",
        ],
        "model-00002-of-00005.safetensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ],
        "model-00003-of-00005.safetensors": [
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
        ],
        "model-00004-of-00005.safetensors": [
            "model.norm.weight",
            "lm_head.weight",
        ],
        "model-00005-of-00005.safetensors": [
            "model.vocab_ids",
            "model.attention_mask",
            "special.测试",
            "special.🚀",
        ],
    }
    
    # 构建 weight_map
    weight_map = {}
    for filename, keys in shard_assignment.items():
        for key in keys:
            weight_map[key] = filename
    
    # 写入各分片
    for filename, keys in shard_assignment.items():
        tensors = {k: all_tensors[k] for k in keys}
        filepath = os.path.join(output_dir, filename)
        metadata = {"shard": filename, "generator": "nn-kit-test"}
        save_file(tensors, filepath, metadata=metadata)
        print(f"  📄 {filename}: {len(tensors)} tensors")
    
    # 写入 index.json
    index = {
        "metadata": {
            "total_size": 0,
            "framework": "pytorch",
        },
        "weight_map": weight_map
    }
    
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"  📜 model.safetensors.index.json: {len(weight_map)} entries")

# ============================================================================
# 9. 无效文件 - 空文件
# ============================================================================

def generate_invalid_empty(output_dir: str):
    """空文件"""
    print("⚠️  Generating: empty.safetensors")
    filepath = os.path.join(output_dir, "empty.safetensors")
    with open(filepath, "wb") as f:
        pass  # 写入 0 字节

# ============================================================================
# 10. 无效文件 - Header 截断
# ============================================================================

def generate_invalid_truncated_header(output_dir: str):
    """Header size 声称很大，但实际数据不够"""
    print("⚠️  Generating: truncated_header.safetensors")
    filepath = os.path.join(output_dir, "truncated_header.safetensors")
    
    # 声称 header 有 1000000 字节，但文件只有几个字节
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', 1000000))  # header size = 1000000
        f.write(b'{"a":')  # 截断的 JSON

# ============================================================================
# 11. 无效文件 - 错误的前 8 字节
# ============================================================================

def generate_invalid_bad_header_size(output_dir: str):
    """Header size 为负数或过大"""
    print("⚠️  Generating: bad_header_size.safetensors")
    filepath = os.path.join(output_dir, "bad_header_size.safetensors")
    
    # Header size = 0xFFFFFFFFFFFFFFFF (max u64)
    with open(filepath, "wb") as f:
        f.write(b'\xff\xff\xff\xff\xff\xff\xff\xff')
        f.write(b'{}')

# ============================================================================
# 12. 无效文件 - JSON 解析错误
# ============================================================================

def generate_invalid_bad_json(output_dir: str):
    """Header 不是有效的 JSON"""
    print("⚠️  Generating: bad_json.safetensors")
    filepath = os.path.join(output_dir, "bad_json.safetensors")
    
    bad_json = b'{"tensor": invalid json here}'
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(bad_json)))
        f.write(bad_json)

# ============================================================================
# 13. 无效文件 - 错误的 offset
# ============================================================================

def generate_invalid_bad_offset(output_dir: str):
    """Tensor 的 data_offsets 超出文件范围"""
    print("⚠️  Generating: bad_offset.safetensors")
    filepath = os.path.join(output_dir, "bad_offset.safetensors")
    
    # 声称 tensor 在 [0, 1000000) 但实际没有那么多数据
    header = json.dumps({
        "tensor": {
            "dtype": "F32",
            "shape": [100, 100],
            "data_offsets": [0, 1000000]  # 需要 1MB 数据
        }
    }).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        f.write(b'\x00' * 100)  # 只有 100 字节数据

# ============================================================================
# 14. 无效文件 - 数据不完整
# ============================================================================

def generate_invalid_incomplete_data(output_dir: str):
    """Header 正确，但数据区截断"""
    print("⚠️  Generating: incomplete_data.safetensors")
    filepath = os.path.join(output_dir, "incomplete_data.safetensors")
    
    # 4x4 float32 = 64 bytes，但我们只写入 32 bytes
    header = json.dumps({
        "tensor": {
            "dtype": "F32",
            "shape": [4, 4],
            "data_offsets": [0, 64]
        }
    }).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        f.write(b'\x00' * 32)  # 只有一半数据

# ============================================================================
# 15. 无效文件 - 不支持的 dtype
# ============================================================================

def generate_invalid_bad_dtype(output_dir: str):
    """使用不存在的 dtype"""
    print("⚠️  Generating: bad_dtype.safetensors")
    filepath = os.path.join(output_dir, "bad_dtype.safetensors")
    
    header = json.dumps({
        "tensor": {
            "dtype": "FLOAT128",  # 不存在的类型
            "shape": [2, 2],
            "data_offsets": [0, 64]
        }
    }).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        f.write(b'\x00' * 64)

# ============================================================================
# 16. 无效文件 - 缺少必要字段
# ============================================================================

def generate_invalid_missing_fields(output_dir: str):
    """Tensor entry 缺少必要字段"""
    print("⚠️  Generating: missing_fields.safetensors")
    filepath = os.path.join(output_dir, "missing_fields.safetensors")
    
    header = json.dumps({
        "tensor": {
            "dtype": "F32",
            # 缺少 shape 和 data_offsets
        }
    }).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)

# ============================================================================
# 17. 边界情况 - 只有 metadata
# ============================================================================

def generate_edge_only_metadata(output_dir: str):
    """只有 __metadata__，没有任何 tensor"""
    print("📦 Generating: only_metadata.safetensors")
    
    # safetensors 库不允许空 tensor dict，我们手动构造
    filepath = os.path.join(output_dir, "only_metadata.safetensors")
    
    header = json.dumps({
        "__metadata__": {
            "info": "This file has no tensors, only metadata"
        }
    }).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        # 没有数据区

# ============================================================================
# 18. 无效分片 - index.json 指向不存在的文件
# ============================================================================

def generate_invalid_sharded_missing_shard(output_dir: str):
    """index.json 引用不存在的分片文件"""
    print("⚠️  Generating: sharded/missing_shard/")
    
    ensure_dir(output_dir)
    
    # 只创建 index.json，不创建实际分片文件
    index = {
        "metadata": {},
        "weight_map": {
            "tensor1": "nonexistent-shard.safetensors",
            "tensor2": "also-missing.safetensors",
        }
    }
    
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

# ============================================================================
# 19. 无效分片 - index.json 格式错误
# ============================================================================

def generate_invalid_sharded_bad_index(output_dir: str):
    """index.json 格式不正确"""
    print("⚠️  Generating: sharded/bad_index/")
    
    ensure_dir(output_dir)
    
    # 创建结构不正确的 index.json
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        f.write('{"weight_map": "should be object not string"}')

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("🚀 NN-Kit Safetensors 测试文件生成器")
    print(f"   输出目录: {OUTPUT_DIR}")
    
    clean_and_create(OUTPUT_DIR)
    
    # ========== 有效文件 ==========
    print_section("1. 生成有效单文件")
    single_dir = os.path.join(OUTPUT_DIR, "valid", "single")
    ensure_dir(single_dir)
    
    generate_single_basic(single_dir)
    generate_single_dtypes(single_dir)
    generate_single_shapes(single_dir)
    generate_single_names(single_dir)
    generate_single_metadata(single_dir)
    generate_single_large(single_dir)
    
    print_section("2. 生成有效分片文件")
    sharded_dir = os.path.join(OUTPUT_DIR, "valid", "sharded")
    ensure_dir(sharded_dir)
    
    generate_sharded_small(os.path.join(sharded_dir, "small"))
    generate_sharded_complex(os.path.join(sharded_dir, "complex"))
    
    # ========== 无效文件 ==========
    print_section("3. 生成无效文件 (用于错误处理测试)")
    invalid_dir = os.path.join(OUTPUT_DIR, "invalid")
    ensure_dir(invalid_dir)
    
    generate_invalid_empty(invalid_dir)
    generate_invalid_truncated_header(invalid_dir)
    generate_invalid_bad_header_size(invalid_dir)
    generate_invalid_bad_json(invalid_dir)
    generate_invalid_bad_offset(invalid_dir)
    generate_invalid_incomplete_data(invalid_dir)
    generate_invalid_bad_dtype(invalid_dir)
    generate_invalid_missing_fields(invalid_dir)
    generate_edge_only_metadata(invalid_dir)
    
    # 无效分片
    generate_invalid_sharded_missing_shard(os.path.join(invalid_dir, "sharded", "missing_shard"))
    generate_invalid_sharded_bad_index(os.path.join(invalid_dir, "sharded", "bad_index"))
    
    # ========== 完成 ==========
    print_section("完成")
    print("✅ 所有测试文件已生成!")
    print(f"   有效文件: {OUTPUT_DIR}/valid/")
    print(f"   无效文件: {OUTPUT_DIR}/invalid/")
    print("")
    print("📋 测试文件清单:")
    
    # 列出所有生成的文件
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            print(f"{sub_indent}{file} ({size_str})")

if __name__ == "__main__":
    main()
