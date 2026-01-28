/**
 * NPY 公开 API
 */

export type { NpyDescriptor } from './types';
export { parseNpy } from './parser';

import { ByteSource } from '../source/types';
import { createByteSource } from '../source';
import { NpyDescriptor } from './types';
import { parseNpy } from './parser';

/**
 * 加载 NPY 文件
 * 
 * @param source - URL、路径、ArrayBuffer 或 File
 * @param signal - 可选的取消信号
 * @returns NpyDescriptor（只包含 metadata，不包含数据）
 * 
 * @example
 * const desc = await loadNpy('./weights.npy');
 * console.log(desc.shape, desc.dtype);
 * const tensor = await Tensor.fromNpy(desc);
 */
export async function loadNpy(
    source: string | ArrayBuffer | File,
    signal?: AbortSignal
): Promise<NpyDescriptor> {
    const byteSource = createByteSource(source);
    return parseNpy(byteSource, signal);
}
