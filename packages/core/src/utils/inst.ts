import { TensorHandleSymbol } from "@kandle/types";

export function ensureIsTensor(input: any, name?: string): void {

    if (input?._handle?.[TensorHandleSymbol] !== true) {
        const msg = name ? `Input to '${name}' must be an instance of Tensor.` : "Input must be an instance of Tensor.";
        throw new Error(msg);
    }

}
