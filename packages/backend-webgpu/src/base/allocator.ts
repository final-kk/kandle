import { IAllocator } from "@kandle/types";
import { WebGPUDeviceManager } from "./device";

export class WebGPUAllocator {

    static alloc(byteSize: number): GPUBuffer {

        const device = WebGPUDeviceManager.device;

        return device.createBuffer({
            size: WebGPUAllocator.alignSize(byteSize),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        })

    }

    static free(buffer: GPUBuffer): void {
        buffer.destroy();
    }

    private static alignSize(size: number): number {

        return Math.ceil(size / 4) * 4;

    }

}
