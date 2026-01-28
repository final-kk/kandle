/**
 * NPY 文件类型定义
 * 
 * NPY is NumPy's native binary format for storing arrays
 */

import { DType } from '@kandle/types';
import { ByteSource } from '../source/types';

/**
 * NPY 文件描述符
 * 
 * 注意: 不持有数据，只持有定位信息
 */
export interface NpyDescriptor {
    /** 转换后的 NN-Kit dtype */
    readonly dtype: DType;

    /** 形状 */
    readonly shape: readonly number[];

    /** 是否 Fortran 顺序 (列优先) */
    readonly fortranOrder: boolean;

    /** 底层数据源 */
    readonly source: ByteSource;

    /** 数据区起始偏移 (header 之后) */
    readonly dataOffset: number;

    /** 数据大小（字节） */
    readonly byteSize: number;

    /** 元素数量 */
    readonly numel: number;

    /** 原始 numpy dtype 字符串 (e.g., '<f4', '>i8') */
    readonly originalDtype: string;

    /** 字节序: 'little' | 'big' */
    readonly byteOrder: 'little' | 'big';
}
