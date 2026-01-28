import { env } from "@kandle/core";
import { WebGPUBackend } from "@kandle/backend-webgpu";

export async function initWebGPU() {
    const backend = await WebGPUBackend.create();
    env.setBackend(backend);
    env.setDefaultDevice(backend.name);
}
