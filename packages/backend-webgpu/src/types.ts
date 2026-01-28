export type WgslDType =
    | "f32"         // 32-bit float (标准)
    | "i32"         // 32-bit signed int (标准)
    | "u32"         // 32-bit unsigned int (标准)
    | "f16"         // 16-bit float (需要 'shader-f16' 扩展)
    | "bool"        // boolean (通常仅用于逻辑变量，存储通常用 u32)
    | "vec2<f32>";  // 用于模拟复数
