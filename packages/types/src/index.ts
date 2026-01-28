export * from "./audio";
export * from "./backend";
export * from "./base";
export * from "./nn";
export * from "./vision";
export * from "./allocator";
export * from "./storage";
export * from "./tensor";
export * from "./random";

// ============================================================================
// Operator Schema v5 (Single Source of Truth)
// ============================================================================

// Re-export the entire opschema module as a namespace
export * as opschema from "./opschema";

// Also export key types directly for convenience
// Also export key types directly for convenience
export type {
    // v5 Core Types
    OpEntry,
    OpSignature,
    ParamDef,
    ReturnDef,
    IteratorConfig,
    KernelConfig,
    CodegenConfig,
    OpName,

    // Pattern & Category
    OpMechanism,
    SchemaDTypeCategory,

    // Value types
    ValueType,

    // Inference rules
    ShapeRule,
    DTypeRule,
} from "./opschema";

// Export helpers and registry
export {
    SchemaT,
    SchemaShape,
    SchemaDtype,
    OpRegistry,
    getAllOpNames,
    getOpEntry,
    getOpVariants,
    getOpsByMechanism,
    MechanismGroups,
} from "./opschema";
