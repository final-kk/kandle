/**
 * Scan Kernels Registration (v5)
 * 
 * Registers all scan (prefix sum) operations with the backend
 * 
 * Supported operations:
 * - cumsum: Cumulative sum along a dimension
 * - cumprod: Cumulative product along a dimension
 * - cummax: Cumulative maximum with indices (returns tuple)
 * - cummin: Cumulative minimum with indices (returns tuple)
 */

import { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executeScan } from './executor';
import { SCAN_OPS } from './ops';

/**
 * Register all scan kernels with the backend
 */
export function registerScanKernels(registry: IBackendOpsRegister): void {
    // Register all scan operations from SCAN_OPS
    for (const dispatchKey of Object.keys(SCAN_OPS)) {
        registry.register(dispatchKey, (iter: ITensorIterator) => {
            executeScan(iter, dispatchKey);
        });
    }
}

// Export all public APIs
export { SCAN_OPS } from './ops';
export { executeScan } from './executor';
export type { ScanOpConfig, ScanDimParams, ScanStrategy } from './types';
export { SCAN_SINGLE_PASS_THRESHOLD } from './types';
