import { IBackendOpsRegister, KernelImpl } from "@kandle/types";

export class SimpleOperatorRegistry implements IBackendOpsRegister {
    private kernels = new Map<string, KernelImpl>();

    register(opName: string, kernelFunc: KernelImpl): void {
        this.kernels.set(opName, kernelFunc);
    }

    find(opName: string): KernelImpl | undefined {
        return this.kernels.get(opName);
    }

    has(opName: string): boolean {
        return this.kernels.has(opName);
    }
}
