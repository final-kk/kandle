import { IBackendOpsRegister, ITensorHandle } from "@kandle/types";
import { MatmulDispatchResult } from "./types";
import { matmulExecutor } from "./executor";

export function registerMatmulOperators(registry: IBackendOpsRegister): void {
    // 注册 matmul kernel
    // 注意：这里的 kernel 接收 (config, inputA, inputB) 而不是 TensorIterator
    registry.register('matmul', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    // mm 和 bmm 使用相同的底层实现，在 dispatcher 层区分
    registry.register('mm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    registry.register('bmm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);
}
