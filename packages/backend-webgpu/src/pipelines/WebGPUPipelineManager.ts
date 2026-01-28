export class WebGPUPipelineManager {

    /**
     * {opname/base.sub}-{dtype}-{fast | general}-{rank_n?}
     * e.g.
     * binary.add-f32-general-3
     * reduction.sum-f32-fast
     * conv2d-f32-general
     * matmul-f32-general-4
     */
    private static pipelines: Map<string, GPUComputePipeline> = new Map<string, GPUComputePipeline>();

    /**
     * BindGroupLayout 缓存
     * e.g.
     * binary.default-layout
     * copy.default-layout
     */
    private static bindGroupLayouts: Map<string, GPUBindGroupLayout> = new Map<string, GPUBindGroupLayout>();


    static getPipeline(key: string): GPUComputePipeline | undefined {

        const pipeline = WebGPUPipelineManager.pipelines.get(key);

        return pipeline;

    }

    static registerPipeline(key: string, pipeline: GPUComputePipeline) {

        if (WebGPUPipelineManager.pipelines.has(key)) {
            console.warn(`Pipeline with key ${key} already exists. Overwriting.`);
        }

        WebGPUPipelineManager.pipelines.set(key, pipeline);
    }

    static getBindGroupLayout(key: string): GPUBindGroupLayout | undefined {
        return WebGPUPipelineManager.bindGroupLayouts.get(key);
    }

    static registerBindGroupLayout(key: string, layout: GPUBindGroupLayout) {
        if (WebGPUPipelineManager.bindGroupLayouts.has(key)) {
            console.warn(`BindGroupLayout with key ${key} already exists. Overwriting.`);
        }
        WebGPUPipelineManager.bindGroupLayouts.set(key, layout);
    }

}

