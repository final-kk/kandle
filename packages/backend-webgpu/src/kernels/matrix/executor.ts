/**
 * MatMul Executor
 * 
 * 执行 tiled matrix multiplication kernel
 */

import { ITensorHandle } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { selectTileConfig, calculateDispatchDimensions } from './tileSelector';
import { buildMmShader, buildBmmShader } from './builder';
import { MatmulDispatchResult, computeGemmVariant, computeMmCacheKey, computeBmmCacheKey, computeDotCacheKey, computeMvCacheKey } from './types';
import {
    MM_UNIFORM_LAYOUT,
    MM_GEMM_UNIFORM_LAYOUT,
    BMM_UNIFORM_LAYOUT,
    BMM_GEMM_UNIFORM_LAYOUT,
    createMmUniformBuffer,
    createMmGemmUniformBuffer,
    createBmmUniformBuffer,
    createBmmGemmUniformBuffer,
} from './uniformLayouts';
import { UniformBufferPool } from '../../base/UniformBufferPool';
import {
    buildDotShader,
    buildMvShader,
    selectDotTileConfig,
    selectMvTileConfig,
    calculateDotDispatch,
    calculateMvDispatch,
    createDotUniformBuffer,
    createMvUniformBuffer,
    MvShaderConfig,
    DotShaderConfig,
} from './specializedKernels';

const logger = new Logger('Matmul-Executor');

// ============================================================
// Executor
// ============================================================

/**
 * 主执行函数
 */
export function matmulExecutor(
    config: MatmulDispatchResult,
    inputA: ITensorHandle,
    inputB: ITensorHandle
): void {
    const { variant, M, K, N, batchSize } = config;

    logger.debug(`Executing matmul: variant=${variant}, M=${M}, K=${K}, N=${N}, batch=${batchSize}`);

    switch (variant) {
        case 'dot':
            executeDotSpecialized(config, inputA, inputB);
            break;
        case 'mv':
            executeMvSpecialized(config, inputA, inputB);
            break;
        case 'mm':
            // 使用 BMM 路径处理 MM，因为:
            // 1. BMM 支持完整的 strided 内存访问
            // 2. batchSize=1 的 BMM 本质上等价于 MM
            // 3. 避免代码重复
            executeBmm(config, inputA, inputB);
            break;
        case 'bmm':
            executeBmm(config, inputA, inputB);
            break;
        default:
            throw new Error(`Unknown matmul variant: ${variant}`);
    }
}

/**
 * 专用 Dot product 执行 (1D @ 1D → scalar) - 工业级 Strided 支持
 * 
 * 使用 tree reduction 实现高效并行归约
 * 支持非连续输入：通过 strides 计算物理地址
 */
function executeDotSpecialized(
    config: MatmulDispatchResult,
    inputA: ITensorHandle,
    inputB: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;
    const { K, computeDtype, output } = config;

    // 工业级：获取输入 strides
    // 对于 1D tensor，stride 就是 strides[0]
    const strideA = inputA.strides.length > 0 ? inputA.strides[inputA.strides.length - 1] : 1;
    const strideB = inputB.strides.length > 0 ? inputB.strides[inputB.strides.length - 1] : 1;

    // 1. 选择 Tile 配置
    const tileConfig = selectDotTileConfig(K);
    logger.debug(`Dot specialized kernel: K=${K}, strideA=${strideA}, strideB=${strideB}, workgroupSize=${tileConfig.workgroupSize}`);

    // 2. 生成或获取 Pipeline (cache key 包含 strides)
    const pipelineKey = computeDotCacheKey(computeDtype, K) + `-s${strideA}-${strideB}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // 3. BindGroupLayout
    const bindGroupLayoutKey = 'matmul.dot-layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(bindGroupLayoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        WebGPUPipelineManager.registerBindGroupLayout(bindGroupLayoutKey, bindGroupLayout);
    }

    // 4. 如果 pipeline 不存在，生成 shader (包含 strides)
    if (!pipeline) {
        const shaderConfig: DotShaderConfig = {
            K,
            dtype: computeDtype,
            strideA,
            strideB,
        };
        const shaderCode = buildDotShader(shaderConfig, tileConfig);
        logger.debug('Generated specialized Dot shader (strided)');

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 5. 准备 Uniform Buffer
    // 注意: offset 已经是元素单位，不需要转换
    const offsetA = inputA.offset;
    const offsetB = inputB.offset;
    const offsetOut = output.offset;

    const uniformArray = createDotUniformBuffer({ K, offsetA, offsetB, offsetOut });

    const uniformBuffer = UniformBufferPool.getInstance().acquire(16, uniformArray);

    // 6. 创建 BindGroup
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputA.buffer as GPUBuffer } },
            { binding: 2, resource: { buffer: inputB.buffer as GPUBuffer } },
            { binding: 3, resource: { buffer: output.buffer as GPUBuffer } },
        ],
    });

    // 7. Dispatch
    const [dispatchX, dispatchY, dispatchZ] = calculateDotDispatch();
    logger.debug(`Dot Dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]`);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 专用 Matrix-Vector 执行 (2D @ 1D → 1D) - 工业级 Strided 支持
 * 
 * 使用 1D workgroup，每个线程处理一行
 * 支持非连续输入：通过 strides 计算物理地址
 */
function executeMvSpecialized(
    config: MatmulDispatchResult,
    inputA: ITensorHandle,
    inputB: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;
    const { M, K, transposeA, computeDtype, output, alpha, beta, inputC } = config;

    // 工业级：获取输入 strides
    // A 是 2D: shape [M, K]，strides [stride_row, stride_col]
    // transposeA 时：逻辑上 A^T 的 [M, K] 对应物理 A 的 [K, M]
    // 所以需要交换 row/col stride
    let strideA_row: number, strideA_col: number;
    if (transposeA) {
        // A^T: 逻辑行 = 物理列，逻辑列 = 物理行
        strideA_row = inputA.strides[inputA.strides.length - 1];  // 沿逻辑 M 移动 = 沿物理 N 移动
        strideA_col = inputA.strides[inputA.strides.length - 2];  // 沿逻辑 K 移动 = 沿物理 M 移动
    } else {
        // A: 正常
        strideA_row = inputA.strides[inputA.strides.length - 2];  // 沿 M 移动
        strideA_col = inputA.strides[inputA.strides.length - 1];  // 沿 K 移动
    }

    // B 是 1D: shape [K]，stride [stride]
    const strideB = inputB.strides.length > 0 ? inputB.strides[inputB.strides.length - 1] : 1;

    // C 是 1D: shape [M]，stride [stride] (如果有)
    const strideC = inputC && inputC.strides.length > 0 ? inputC.strides[inputC.strides.length - 1] : 1;

    // 1. 选择 Tile 配置
    const tileConfig = selectMvTileConfig(M, K);
    logger.debug(`MV specialized kernel: M=${M}, K=${K}, strideA=[${strideA_row},${strideA_col}], strideB=${strideB}, workgroupSize=${tileConfig.workgroupSize}`);

    // 2. 确定是否有 GEMM
    const hasInputC = inputC !== undefined && beta !== 0.0;

    // 3. 生成或获取 Pipeline (cache key 包含 strides)
    const gemmVariant = computeGemmVariant(alpha, beta, hasInputC);
    const pipelineKey = computeMvCacheKey(computeDtype, M, K, transposeA) + `-${gemmVariant}-s${strideA_row}-${strideA_col}-${strideB}-${strideC}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // 4. BindGroupLayout
    const bindGroupLayoutKey = hasInputC ? 'matmul.mv-gemm-layout' : 'matmul.mv-layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(bindGroupLayoutKey);

    if (!bindGroupLayout) {
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ];
        if (hasInputC) {
            entries.push({ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
        }
        bindGroupLayout = device.createBindGroupLayout({ entries });
        WebGPUPipelineManager.registerBindGroupLayout(bindGroupLayoutKey, bindGroupLayout);
    }

    // 5. 如果 pipeline 不存在，生成 shader (包含 strides)
    if (!pipeline) {
        const shaderConfig: MvShaderConfig = {
            M, K, transposeA, dtype: computeDtype, alpha, beta, hasInputC,
            strideA_row,
            strideA_col,
            strideB,
            strideC,
        };
        const shaderCode = buildMvShader(shaderConfig, tileConfig);
        logger.debug('Generated specialized MV shader (strided)');

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 6. 准备 Uniform Buffer
    // 注意: offset 已经是元素单位，不需要转换
    const offsetA = inputA.offset;
    const offsetB = inputB.offset;
    const offsetOut = output.offset;
    const offsetC = hasInputC && inputC ? inputC.offset : 0;

    const uniformArray = createMvUniformBuffer({
        M, K, offsetA, offsetB, offsetC, offsetOut, alpha, beta
    }, hasInputC);

    const uniformBuffer = UniformBufferPool.getInstance().acquire(32, uniformArray);

    // 7. 创建 BindGroup
    const bindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputA.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: inputB.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: output.buffer as GPUBuffer } },
    ];
    if (hasInputC && inputC) {
        bindGroupEntries.push({ binding: 4, resource: { buffer: inputC.buffer as GPUBuffer } });
    }

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries,
    });

    // 8. Dispatch
    const [dispatchX, dispatchY, dispatchZ] = calculateMvDispatch(M, tileConfig);
    logger.debug(`MV Dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]`);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Matrix-Matrix 执行 (2D @ 2D → 2D)
 */
function executeMm(
    config: MatmulDispatchResult,
    inputA: ITensorHandle,
    inputB: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;
    const { M, K, N, transposeA, transposeB, computeDtype, output, alpha, beta, inputC } = config;

    // 1. 选择 Tile 配置
    const tileConfig = selectTileConfig(M, K, N, computeDtype);
    logger.debug(`Tile config: vec4=${tileConfig.useVec4}, workPerThread=${tileConfig.workPerThread}, wgSize=${tileConfig.workgroupSize}`);

    // 2. 确定 GEMM 模式（用于 shader 变体选择）
    const hasInputC = inputC !== undefined && beta !== 0.0;
    const gemmVariant = computeGemmVariant(alpha, beta, hasInputC);

    // 3. 生成或获取 Pipeline（使用配置化缓存键）
    const pipelineKey = computeMmCacheKey(
        computeDtype, M, K, N, transposeA, transposeB, tileConfig.useVec4, gemmVariant
    );
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // 4. 创建或获取 BindGroupLayout
    // 根据是否有 inputC 选择不同的 layout
    const bindGroupLayoutKey = hasInputC ? 'matmul.gemm-layout' : 'matmul.default-layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(bindGroupLayoutKey);

    if (!bindGroupLayout) {
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // uniforms
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // inputA
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // inputB
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
        ];

        if (hasInputC) {
            entries.push(
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } } // inputC
            );
        }

        bindGroupLayout = device.createBindGroupLayout({ entries });
        WebGPUPipelineManager.registerBindGroupLayout(bindGroupLayoutKey, bindGroupLayout);
    }

    // 5. 如果 pipeline 不存在，生成 shader 并创建 pipeline
    if (!pipeline) {
        const shaderCode = buildMmShader(config, tileConfig);
        logger.debug('Shader code generated for GEMM');

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 6. 准备 Uniform Buffer
    // 注意: offset 已经是元素单位，不需要转换
    const offsetA = inputA.offset;
    const offsetB = inputB.offset;
    const offsetOut = output.offset;
    const offsetC = hasInputC && inputC ? inputC.offset : 0;

    // 获取 InputC 的形状用于广播
    let cShapeM = 1;
    let cShapeN = 1;
    if (hasInputC && inputC) {
        const cShape = inputC.shape;
        if (cShape.length === 0) {
            cShapeM = 1;
            cShapeN = 1;
        } else if (cShape.length === 1) {
            cShapeM = 1;
            cShapeN = cShape[0];
        } else {
            cShapeM = cShape[cShape.length - 2];
            cShapeN = cShape[cShape.length - 1];
        }
    }

    // 使用结构化 Uniform Buffer
    const uniformLayout = hasInputC ? MM_GEMM_UNIFORM_LAYOUT : MM_UNIFORM_LAYOUT;
    const uniformArray = hasInputC
        ? createMmGemmUniformBuffer({
            M, K, N,
            offsetA, offsetB, offsetC, offsetOut,
            cShapeM, cShapeN,
            alpha, beta
        })
        : createMmUniformBuffer({
            M, K, N,
            offsetA, offsetB, offsetOut,
            alpha, beta
        });

    const uniformBuffer = UniformBufferPool.getInstance().acquire(uniformLayout.size, uniformArray);

    // 7. 创建 BindGroup
    const bindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputA.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: inputB.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: output.buffer as GPUBuffer } },
    ];

    if (hasInputC && inputC) {
        bindGroupEntries.push(
            { binding: 4, resource: { buffer: inputC.buffer as GPUBuffer } }
        );
    }

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries,
    });

    // 8. Dispatch
    const [dispatchX, dispatchY, dispatchZ] = calculateDispatchDimensions(M, N, 1, tileConfig);
    logger.debug(`Dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}], GEMM variant: ${gemmVariant}`);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Batched Matrix-Matrix 执行 (≥3D @ ≥3D → ≥3D)
 */
function executeBmm(
    config: MatmulDispatchResult,
    inputA: ITensorHandle,
    inputB: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;
    const { M, K, N, batchSize, batchShape, transposeA, transposeB, computeDtype, output, alpha, beta, inputC } = config;

    // 1. 选择 Tile 配置
    const tileConfig = selectTileConfig(M, K, N, computeDtype);

    // 2. 获取 batch 形状
    const batchShapeA = inputA.shape.slice(0, -2);
    const batchShapeB = inputB.shape.slice(0, -2);

    // 3. 确定 GEMM 模式（使用配置化缓存键）
    const hasInputC = inputC !== undefined && beta !== 0.0;
    const gemmVariant = computeGemmVariant(alpha, beta, hasInputC);
    const batchShapeC = hasInputC && inputC ? inputC.shape.slice(0, -2) : undefined;

    // 4. 生成或获取 Pipeline
    const pipelineKey = computeBmmCacheKey(
        computeDtype, M, K, N, batchSize, transposeA, transposeB,
        batchShapeA, batchShapeB, batchShapeC, gemmVariant
    );
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // 5. 创建或获取 BindGroupLayout
    const bindGroupLayoutKey = hasInputC ? 'matmul.gemm-layout' : 'matmul.default-layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(bindGroupLayoutKey);

    if (!bindGroupLayout) {
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ];

        if (hasInputC) {
            entries.push(
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
            );
        }

        bindGroupLayout = device.createBindGroupLayout({ entries });
        WebGPUPipelineManager.registerBindGroupLayout(bindGroupLayoutKey, bindGroupLayout);
    }

    // 6. 如果 pipeline 不存在，生成 shader
    if (!pipeline) {
        // 获取 InputC 的 batch shape（如果存在）
        const batchShapeC = hasInputC && inputC ? inputC.shape.slice(0, -2) : undefined;

        const shaderCode = buildBmmShader(config, tileConfig, batchShapeA, batchShapeB, batchShapeC);
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 7. 准备 Uniform Buffer
    // 注意: offset 已经是元素单位，不需要转换
    const offsetA = inputA.offset;
    const offsetB = inputB.offset;
    const offsetOut = output.offset;
    const offsetC = hasInputC && inputC ? inputC.offset : 0;

    // 获取 InputC 的 M/N 形状用于广播
    let cShapeM = 1;
    let cShapeN = 1;
    if (hasInputC && inputC) {
        const cShape = inputC.shape;
        if (cShape.length === 0) {
            cShapeM = 1;
            cShapeN = 1;
        } else if (cShape.length === 1) {
            cShapeM = 1;
            cShapeN = cShape[0];
        } else {
            cShapeM = cShape[cShape.length - 2];
            cShapeN = cShape[cShape.length - 1];
        }
    }

    // 使用结构化 Uniform Buffer
    const uniformLayout = hasInputC ? BMM_GEMM_UNIFORM_LAYOUT : BMM_UNIFORM_LAYOUT;

    // 从 config 获取 strides 信息 (legacy)
    const [strideARow, strideACol] = config.stridesA;
    const [strideBRow, strideBCol] = config.stridesB;

    // 获取完整 4D strides (真正的 4D 支持)
    const fullStridesA = config.fullStridesA as readonly [number, number, number, number];
    const fullStridesB = config.fullStridesB as readonly [number, number, number, number];

    const uniformArray = hasInputC
        ? createBmmGemmUniformBuffer({
            M, K, N, batchSize,
            offsetA, offsetB, offsetC, offsetOut,
            ndimA: config.ndimA,
            ndimB: config.ndimB,
            fullStridesA,
            fullStridesB,
            strideARow, strideACol,
            strideBRow, strideBCol,
            batchStrideA: config.batchStrideA,
            batchStrideB: config.batchStrideB,
            cShapeM, cShapeN,
            alpha, beta
        })
        : createBmmUniformBuffer({
            M, K, N, batchSize,
            offsetA, offsetB, offsetOut,
            ndimA: config.ndimA,
            ndimB: config.ndimB,
            fullStridesA,
            fullStridesB,
            strideARow, strideACol,
            strideBRow, strideBCol,
            batchStrideA: config.batchStrideA,
            batchStrideB: config.batchStrideB,
            alpha, beta
        });

    const uniformBuffer = UniformBufferPool.getInstance().acquire(uniformLayout.size, uniformArray);

    // 8. 创建 BindGroup
    const bindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputA.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: inputB.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: output.buffer as GPUBuffer } },
    ];

    if (hasInputC && inputC) {
        bindGroupEntries.push(
            { binding: 4, resource: { buffer: inputC.buffer as GPUBuffer } }
        );
    }

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries,
    });

    // 9. Dispatch
    const [dispatchX, dispatchY, _] = calculateDispatchDimensions(M, N, batchSize, tileConfig);
    const dispatchZ = batchSize;
    logger.debug(`BMM Dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}], GEMM variant: ${gemmVariant}`);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

