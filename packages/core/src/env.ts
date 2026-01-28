import { DeviceNameEnum, IBackend } from "@kandle/types";

class Environment {

    private _backends: Partial<Record<DeviceNameEnum, IBackend>> = {};
    private _defaultDevice: DeviceNameEnum = DeviceNameEnum.JS;

    setBackend(backend: IBackend) {
        if (this._backends[backend.name]) {
            // throw new Error(`Backend ${backend.name} is already registered.`);
            console.warn(`Backend ${backend.name} is already registered.`);
            return
        }

        this._backends[backend.name] = backend;
    }

    getBackend(name: DeviceNameEnum): IBackend {
        const backend = this._backends[name];

        if (!backend) {
            throw new Error(`Backend ${name} is not registered.`);
        }

        return backend;
    }

    setDefaultDevice(device: DeviceNameEnum) {
        this._defaultDevice = device;
    }

    getDefaultDevice(): IBackend {

        const backend = this._backends[this._defaultDevice];

        if (!backend) {
            throw new Error(`Backend ${this._defaultDevice} is not registered.`);
        }

        return backend;
    }
}

export const env = new Environment();