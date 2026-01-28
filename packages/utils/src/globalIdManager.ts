export class GlobalIdManager {
    private static _globalTensorIdCounter = 0;
    private static _globalStorageIdCounter = 0;

    static getNextTensorId(): number {
        return this._globalTensorIdCounter++;
    }

    static getNextStorageId(): number {
        return this._globalStorageIdCounter++;
    }
}