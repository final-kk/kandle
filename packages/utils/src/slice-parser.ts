import { SliceParam } from "@kandle/types";

/**
 * Parser for string-based tensor slicing (NumPy/Python style).
 */

const INT_MAX = Number.MAX_SAFE_INTEGER;
const INT_MIN = Number.MIN_SAFE_INTEGER;

/**
 * Parses a slice string into explicit start, end, axis, step arrays.
 * 
 * Supports:
 * - Basic slicing: "start:end:step", "start:end", ":end", "start:", ":"
 * - Negative indices: "-1", "-2:"
 * - Step: "::-1"
 * - Ellipsis: "..." (requires shape to be provided)
 * - Integer indexing: "1" (treated as 1:2, does not reduce rank)
 * 
 * @param query The slice string (e.g., "0:10, :, ...")
 * @param shape The shape of the tensor (required for '...' expansion and negative index validation if needed, though C++ handles neg indices)
 */
export function parseSliceString(query: string, shape?: readonly number[]): SliceParam[] {
    // Clean up query: remove brackets if present, trim
    query = query.trim();
    if (query.startsWith('[') && query.endsWith(']')) {
        query = query.slice(1, -1);
    }
    
    if (!query) {
        return [];
    }

    const parts = query.split(',').map(s => s.trim()).filter(s => s.length > 0);
    
    // Temporary storage for parsed parts
    const parsedStarts: number[] = [];
    const parsedEnds: number[] = [];
    const parsedSteps: number[] = [];
    const parsedAxes: number[] = [];

    let ellipsisIndex = -1;
    for (let i = 0; i < parts.length; i++) {
        if (parts[i] === '...') {
            if (ellipsisIndex !== -1) {
                throw new Error("Only one ellipsis '...' is allowed in a slice string.");
            }
            ellipsisIndex = i;
        }
    }

    // Determine axes mapping
    let axisIndices: number[] = [];
    
    if (ellipsisIndex !== -1) {
        if (!shape) {
            throw new Error("Shape is required to resolve ellipsis '...' in slice string.");
        }
        const rank = shape.length;
        const numPartsBefore = ellipsisIndex;
        const numPartsAfter = parts.length - 1 - ellipsisIndex;
        const numEllipsisDims = rank - numPartsBefore - numPartsAfter;

        if (numEllipsisDims < 0) {
            throw new Error(`Too many slice indices for shape [${shape.join(',')}]`);
        }

        // Before ellipsis
        for (let i = 0; i < numPartsBefore; i++) {
            axisIndices.push(i);
        }
        // Skip ellipsis dimensions
        // After ellipsis
        for (let i = 0; i < numPartsAfter; i++) {
            axisIndices.push(rank - numPartsAfter + i);
        }
    } else {
        // No ellipsis: standard 0-indexed mapping
        for (let i = 0; i < parts.length; i++) {
            axisIndices.push(i);
        }
        if (shape && parts.length > shape.length) {
             throw new Error(`Too many slice indices for shape [${shape.join(',')}]`);
        }
    }

    // Filter out '...' from parts to match axisIndices
    const activeParts = parts.filter(p => p !== '...');

    if (activeParts.length !== axisIndices.length) {
        throw new Error("Internal Logic Error: active parts count mismatch");
    }

    // Parse each part
    for (let i = 0; i < activeParts.length; i++) {
        const part = activeParts[i];
        const axis = axisIndices[i];
        
        // Check for integer indexing (no colon)
        if (!part.includes(':')) {
            const idx = parseInt(part, 10);
            if (isNaN(idx)) {
                throw new Error(`Invalid slice index: ${part}`);
            }
            
            // Note on Integer Indexing:
            // This parser converts integer indices (e.g., "1", "-1") into slice ranges (e.g., "1:2", "-1:MAX").
            // This preserves the rank of the tensor.
            
            let start = idx;
            let end = idx + 1;
            if (idx === -1) {
                end = INT_MAX; 
            } else if (idx < -1) {
                // idx = -2. start = -2. end = -1.
            }
            
            parsedStarts.push(start);
            parsedEnds.push(end);
            parsedSteps.push(1);
            parsedAxes.push(axis);
            continue;
        }

        // Parse start:end:step
        const split = part.split(':');
        
        let startStr = split[0];
        let endStr = split[1];
        let stepStr = split.length > 2 ? split[2] : "";

        let step = 1;
        if (stepStr && stepStr.trim().length > 0) {
            step = parseInt(stepStr, 10);
            if (isNaN(step)) throw new Error(`Invalid step: ${stepStr}`);
            if (step === 0) throw new Error("Slice step cannot be zero");
        }

        let start: number;
        let end: number;

        // Default start/end depend on sign of step
        if (step > 0) {
            if (!startStr || startStr.trim().length === 0) {
                start = 0;
            } else {
                start = parseInt(startStr, 10);
            }

            if (!endStr || endStr.trim().length === 0) {
                end = INT_MAX;
            } else {
                end = parseInt(endStr, 10);
            }
        } else {
            if (!startStr || startStr.trim().length === 0) {
                start = INT_MAX; // Maps to dim-1
            } else {
                start = parseInt(startStr, 10);
            }

            if (!endStr || endStr.trim().length === 0) {
                end = INT_MIN; // Maps to -1
            } else {
                end = parseInt(endStr, 10);
            }
        }

        if (isNaN(start)) throw new Error(`Invalid start index: ${startStr}`);
        if (isNaN(end)) throw new Error(`Invalid end index: ${endStr}`);

        parsedStarts.push(start);
        parsedEnds.push(end);
        parsedSteps.push(step);
        parsedAxes.push(axis);
    }

    // Construct final SliceParam array
    if (shape) {
        const specs: SliceParam[] = [];
        // Initialize with full slices for all dimensions
        for (let i = 0; i < shape.length; i++) {
            specs.push({ start: 0, end: shape[i], step: 1 });
        }
        
        // Overwrite with parsed specs
        for (let i = 0; i < parsedAxes.length; i++) {
            const axis = parsedAxes[i];
            const dimSize = shape[axis];
            
            let s = parsedStarts[i];
            let e = parsedEnds[i];
            let st = parsedSteps[i];

            if (st > 0) {
                // Forward slice
                if (e === INT_MAX) e = dimSize;
                
                if (s < 0) s += dimSize;
                if (e < 0) e += dimSize;
                
                // Clamp
                if (s < 0) s = 0;
                if (s > dimSize) s = dimSize;
                if (e < 0) e = 0;
                if (e > dimSize) e = dimSize;
            } else {
                // Backward slice
                if (s === INT_MAX) s = dimSize - 1;
                else if (s < 0) s += dimSize;
                
                if (e === INT_MIN) e = -1 - dimSize;
                else if (e < 0) e += dimSize;
                
                // Clamp
                if (s < 0) s = -dimSize;
                if (s > dimSize - 1) s = dimSize - 1;
                
                if (e < 0) e = -1 - dimSize;
                if (e > dimSize - 1) e = dimSize - 1;
            }

            specs[axis] = {
                start: s,
                end: e,
                step: st
            };
        }
        return specs;
    } else {
        // No shape provided. Return specs in parsed order (which corresponds to axes 0, 1, 2...)
        const specs: SliceParam[] = [];
        for (let i = 0; i < parsedStarts.length; i++) {
             specs.push({
                start: parsedStarts[i],
                end: parsedEnds[i],
                step: parsedSteps[i]
            });
        }
        return specs;
    }
}