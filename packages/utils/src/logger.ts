/**
 * Logger Utility
 * 
 * 可配置的日志工具，支持模块前缀和日志级别控制。
 * 各模块实例化自己的 Logger，打印时带上 [module] 前缀。
 */

export enum LogLevel {
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4,
}

// 全局日志级别
let globalLogLevel: LogLevel = LogLevel.WARN;

/**
 * 设置全局日志级别
 */
export function setGlobalLogLevel(level: LogLevel): void {
    globalLogLevel = level;
}

/**
 * 获取当前全局日志级别
 */
export function getGlobalLogLevel(): LogLevel {
    return globalLogLevel;
}

/**
 * Logger 类
 * 
 * 每个模块实例化自己的 Logger，带上模块名前缀。
 * 
 * @example
 * ```typescript
 * const logger = new Logger('Backend-WebGPU');
 * logger.debug('Pipeline created');
 * // 输出: [Backend-WebGPU] Pipeline created
 * ```
 */
export class Logger {
    private readonly module: string;
    private localLevel?: LogLevel;

    constructor(module: string) {
        this.module = module;
    }

    /**
     * 设置此 Logger 实例的局部日志级别
     * 如果设置，将覆盖全局日志级别
     */
    setLevel(level: LogLevel): void {
        this.localLevel = level;
    }

    /**
     * 获取有效的日志级别
     */
    private getEffectiveLevel(): LogLevel {
        return this.localLevel ?? globalLogLevel;
    }

    /**
     * 格式化前缀
     */
    private prefix(): string {
        return `[${this.module}]`;
    }

    /**
     * DEBUG 级别日志
     */
    debug(...args: any[]): void {
        if (this.getEffectiveLevel() >= LogLevel.DEBUG) {
            console.log(this.prefix(), ...args);
        }
    }

    /**
     * INFO 级别日志
     */
    info(...args: any[]): void {
        if (this.getEffectiveLevel() >= LogLevel.INFO) {
            console.info(this.prefix(), ...args);
        }
    }

    /**
     * WARN 级别日志
     */
    warn(...args: any[]): void {
        if (this.getEffectiveLevel() >= LogLevel.WARN) {
            console.warn(this.prefix(), ...args);
        }
    }

    /**
     * ERROR 级别日志
     */
    error(...args: any[]): void {
        if (this.getEffectiveLevel() >= LogLevel.ERROR) {
            console.error(this.prefix(), ...args);
        }
    }
}
