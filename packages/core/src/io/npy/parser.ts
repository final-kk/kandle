/**
 * NPY 文件解析器
 * 
 * NPY 文件格式:
 * - 6 bytes: magic "\x93NUMPY"
 * - 1 byte: major version
 * - 1 byte: minor version
 * - 2/4 bytes: header len (v1: 2 bytes, v2+: 4 bytes)
 * - N bytes: header dict (Python literal)
 * - padding to 64-byte alignment
 * - rest: data
 */

import { DType } from '@kandle/types';
import { ByteSource } from '../source/types';
import { NpyDescriptor } from './types';
import { getDtypeBytes } from '../safetensor/dtypes';

// ============================================================================
// NPY DType Mapping
// ============================================================================

/**
 * NumPy dtype 字符到 NN-Kit dtype 的映射
 * 
 * NumPy dtype format: [byte order char][type char][byte size]
 * - byte order: '<' (little), '>' (big), '=' (native), '|' (not applicable)
 * - type char: 'f' (float), 'i' (int), 'u' (uint), 'b' (bool), 'c' (complex)
 */
function mapNumpyDtype(descr: string): { dtype: DType; byteOrder: 'little' | 'big' } {
    // 处理字节序
    let byteOrder: 'little' | 'big' = 'little';
    let typeStr = descr;

    if (descr.startsWith('<')) {
        byteOrder = 'little';
        typeStr = descr.slice(1);
    } else if (descr.startsWith('>')) {
        byteOrder = 'big';
        typeStr = descr.slice(1);
    } else if (descr.startsWith('=') || descr.startsWith('|')) {
        // Native or not applicable - assume little endian for web
        byteOrder = 'little';
        typeStr = descr.slice(1);
    }

    // 映射类型
    const mapping: Record<string, DType> = {
        // Floats
        'f2': 'float16',
        'f4': 'float32',
        'f8': 'float64',
        // Signed integers
        'i1': 'int8',
        'i2': 'int16',
        'i4': 'int32',
        'i8': 'int64',
        // Unsigned integers
        'u1': 'uint8',
        'u2': 'uint16',
        'u4': 'uint32',
        'u8': 'uint64',
        // Boolean
        'b1': 'bool',
        // Complex (stored as pairs of floats)
        'c8': 'complex64',
        'c16': 'complex128',
    };

    const dtype = mapping[typeStr];
    if (!dtype) {
        throw new Error(`Unsupported NumPy dtype: ${descr}`);
    }

    return { dtype, byteOrder };
}

// ============================================================================
// Header Parsing
// ============================================================================

/**
 * 解析 Python dict literal (简化版)
 * 
 * 支持格式: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4)}
 */
function parseNpyHeaderDict(headerStr: string): {
    dtype: DType;
    shape: number[];
    fortranOrder: boolean;
    originalDtype: string;
    byteOrder: 'little' | 'big';
} {
    // 移除首尾空白
    headerStr = headerStr.trim();

    // 提取 'descr' 值
    const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
    if (!descrMatch) {
        throw new Error("Invalid NPY header: missing 'descr' field");
    }
    const originalDtype = descrMatch[1];
    const { dtype, byteOrder } = mapNumpyDtype(originalDtype);

    // 提取 'fortran_order' 值
    const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
    const fortranOrder = fortranMatch ? fortranMatch[1] === 'True' : false;

    // 提取 'shape' 值
    const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
    if (!shapeMatch) {
        throw new Error("Invalid NPY header: missing 'shape' field");
    }

    // 解析 shape tuple
    const shapeStr = shapeMatch[1].trim();
    let shape: number[];
    if (shapeStr === '') {
        // 标量: shape = ()
        shape = [];
    } else {
        // 移除末尾逗号 (单元素 tuple: (3,))
        shape = shapeStr
            .replace(/,\s*$/, '')
            .split(',')
            .map(s => parseInt(s.trim(), 10));
    }

    return { dtype, shape, fortranOrder, originalDtype, byteOrder };
}

/**
 * 解析 NPY 文件
 * 
 * @param source - 数据源
 * @param signal - 可选的取消信号
 * @returns NPY 描述符
 */
export async function parseNpy(
    source: ByteSource,
    signal?: AbortSignal
): Promise<NpyDescriptor> {
    // 1. 读取前 12 字节 (magic + version + header_len)
    const preHeader = await source.read(0, 12, signal);
    const view = new DataView(preHeader);
    const bytes = new Uint8Array(preHeader);

    // 2. 验证 magic "\x93NUMPY"
    if (bytes[0] !== 0x93 ||
        bytes[1] !== 0x4E || // 'N'
        bytes[2] !== 0x55 || // 'U'
        bytes[3] !== 0x4D || // 'M'
        bytes[4] !== 0x50 || // 'P'
        bytes[5] !== 0x59) { // 'Y'
        throw new Error('Invalid NPY file: bad magic number');
    }

    // 3. 解析版本
    const majorVersion = bytes[6];
    const minorVersion = bytes[7];

    // 4. 解析 header length
    let headerLen: number;
    let headerStart: number;

    if (majorVersion === 1) {
        // Version 1.0: 2-byte header length (little-endian)
        headerLen = view.getUint16(8, true);
        headerStart = 10;
    } else if (majorVersion >= 2) {
        // Version 2.0+: 4-byte header length (little-endian)
        headerLen = view.getUint32(8, true);
        headerStart = 12;
    } else {
        throw new Error(`Unsupported NPY version: ${majorVersion}.${minorVersion}`);
    }

    // 5. 读取 header dict
    const headerBuffer = await source.read(headerStart, headerLen, signal);
    const headerStr = new TextDecoder('utf-8').decode(headerBuffer);
    const header = parseNpyHeaderDict(headerStr);

    // 6. 计算数据偏移
    const dataOffset = headerStart + headerLen;

    // 7. 计算数据大小
    const numel = header.shape.length === 0 ? 1 : header.shape.reduce((a, b) => a * b, 1);
    const bytesPerElement = getDtypeBytes(header.dtype);
    const byteSize = numel * bytesPerElement;

    return {
        dtype: header.dtype,
        shape: header.shape,
        fortranOrder: header.fortranOrder,
        source,
        dataOffset,
        byteSize,
        numel,
        originalDtype: header.originalDtype,
        byteOrder: header.byteOrder,
    };
}
