import { ITensorHandle } from "./tensor";

export interface IVision {
    load(filePath: string | ArrayBuffer): ITensorHandle;
    save(filePath: string, imageTensor: ITensorHandle): void;
    imdecode(buffer: ArrayBuffer): ITensorHandle;
    resize(image: ITensorHandle, newWidth: number, newHeight: number): ITensorHandle;
    crop(image: ITensorHandle, top: number, left: number, height: number, width: number): ITensorHandle;
    rotate(image: ITensorHandle, angle: number): ITensorHandle;
    flip(image: ITensorHandle, horizontal: boolean, vertical: boolean): ITensorHandle;
    normalize(image: ITensorHandle, mean: number[], std: number[]): ITensorHandle;
}
