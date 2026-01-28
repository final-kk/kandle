/**
 * Whisper 端到端推理示例 - Node.js 版本
 *
 * 使用方法：npm run whisper
 *
 * 工作流程：
 * 1. 初始化 WebGPU 后端
 * 2. 加载 Tokenizer
 * 3. 加载 Whisper 模型权重
 * 4. 加载并预处理音频文件
 * 5. 运行推理生成转录文本
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { Tensor, io } from '@kandle/core';
import {
    WhisperModel,
    type WhisperModelSize,
    getWhisperConfig,
    WHISPER_AUDIO_CONFIG,
    processHFWhisperWeights,
} from '@kandle/model-utils';
import { Tokenizer } from '@huggingface/tokenizers';
import { init } from '../init';
import { WhisperFeatureExtractor } from './audio';
import {
    DEFAULT_MODEL_SIZE,
    EOT_TOKEN_ID,
    logger,
    getTensorStats,
    assert,
} from './utils';

// ============================================================================
// 配置常量 - 在此修改模型路径和推理参数
// ============================================================================

/** 模型尺寸 (可选: tiny, base, small, medium, large-v3, large-v3-turbo) */
const MODEL_SIZE: WhisperModelSize = 'base';

/** 模型根目录（修改为你的模型路径） */
const MODEL_ROOT_DIR = '/models/whisper-base';

/** 派生路径 */
const MODEL_PATH = path.join(MODEL_ROOT_DIR, 'model.safetensors');
const TOKENIZER_JSON_PATH = path.join(MODEL_ROOT_DIR, 'tokenizer.json');

/** 测试音频文件路径 */
const AUDIO_FILE_PATH = path.join(__dirname, 'en.wav');

/** 最大生成 token 数量 */
const MAX_NEW_TOKENS = 224;

/** 语言代码 (如 'en', 'zh', 'ja') */
const LANGUAGE = 'en';

/** 任务类型 ('transcribe' 或 'translate') */
const TASK: 'transcribe' | 'translate' = 'transcribe';

// ============================================================================
// 状态管理
// ============================================================================

let _tokenizer: Tokenizer | null = null;
let _model: WhisperModel | null = null;
let _featureExtractor: WhisperFeatureExtractor | null = null;

// ============================================================================
// 步骤 1: 加载 Tokenizer
// ============================================================================

/**
 * 步骤 1: 加载 Tokenizer
 * 从本地文件系统读取 tokenizer.json
 */
async function loadTokenizer(): Promise<void> {
    logger.logGroup('步骤 1: 加载 Tokenizer');

    try {
        logger.info(`从目录加载: ${MODEL_ROOT_DIR}`);

        // 检查文件是否存在
        await fs.access(TOKENIZER_JSON_PATH).catch(() => {
            throw new Error(
                `Tokenizer 文件不存在。请检查路径配置：\n` +
                `  TOKENIZER_JSON_PATH = ${TOKENIZER_JSON_PATH}`
            );
        });

        // 读取并解析 tokenizer 文件
        const jsonContent = await fs.readFile(TOKENIZER_JSON_PATH, 'utf-8');
        const tokenizerJson = JSON.parse(jsonContent);

        _tokenizer = new Tokenizer(tokenizerJson, {});

        // 验证 tokenizer 功能
        const testEncode = _tokenizer.encode('Hello');
        logger.info(`测试编码 "Hello": [${testEncode.ids.join(', ')}]`);
        logger.success('✓ Tokenizer 加载成功');
    } catch (e) {
        logger.error(`加载 Tokenizer 失败: ${(e as Error).message}`);
        throw e;
    }
}

function getTokenizer(): Tokenizer {
    if (!_tokenizer) throw new Error('Tokenizer not loaded');
    return _tokenizer;
}

// ============================================================================
// 步骤 2: 加载模型
// ============================================================================

/**
 * 加载模型权重（从 safetensors 文件）
 *
 * HuggingFace Whisper 使用分开的 q_proj, k_proj, v_proj 投影，
 * 需要转换为 PyTorch 标准的 in_proj_weight 格式。
 */
async function loadModelWeights(model: WhisperModel): Promise<void> {
    logger.info(`从文件加载权重: ${MODEL_PATH}`);

    try {
        // 检查模型文件是否存在
        await fs.access(MODEL_PATH).catch(() => {
            throw new Error(
                `模型文件不存在: ${MODEL_PATH}\n` +
                `请检查 MODEL_ROOT_DIR 配置是否正确`
            );
        });

        // 读取 safetensors 文件（Node.js Buffer → ArrayBuffer）
        const fileBuffer = await fs.readFile(MODEL_PATH);
        const arrayBuffer = fileBuffer.buffer.slice(
            fileBuffer.byteOffset,
            fileBuffer.byteOffset + fileBuffer.byteLength
        );

        // 加载 SafetensorGroup
        const group = await io.loadSafetensor(arrayBuffer);
        logger.info(`SafeTensor 加载: ${group.layers.size} 层`);

        // ==========================================
        // 步骤 1: 预处理 HF 权重 (合并 q/k/v_proj → in_proj)
        // ==========================================
        logger.info('预处理 HuggingFace 权重格式...');
        const { weights: mergedWeights, processedHFKeys } =
            await processHFWhisperWeights(group, 'model.');
        logger.info(`已合并 ${mergedWeights.size} 个 attention 投影权重`);

        // ==========================================
        // 步骤 2: 使用 keyMapper 加载其他权重
        // ==========================================
        const result = await model.loadFromSafetensor(group, {
            strict: false,
            keyMapper: (key) => {
                // 跳过已处理的 q/k/v_proj 键
                if (processedHFKeys.has(key)) {
                    return '__skip__';  // 返回不存在的键名，会被标记为 unexpected
                }
                // HuggingFace 模型键映射: model.encoder.* → encoder.*
                return key.replace(/^model\./, '');
            },
        });

        // ==========================================
        // 步骤 3: 手动应用合并后的权重
        // ==========================================
        let appliedMerged = 0;
        for (const [name, param] of model.namedParameters('', true)) {
            const mergedWeight = mergedWeights.get(name);
            if (mergedWeight) {
                // 验证形状
                const paramShape = param.shape;
                const weightShape = mergedWeight.shape;
                if (paramShape.join(',') !== weightShape.join(',')) {
                    logger.error(
                        `Shape mismatch for ${name}: ` +
                        `expected [${paramShape}], got [${weightShape}]`
                    );
                    continue;
                }

                // 替换 handle (零拷贝方式)
                (param as any)._handle = (mergedWeight as any)._handle;
                appliedMerged++;
                // 从 map 中移除已应用的权重，避免后续 dispose
                mergedWeights.delete(name);
            }
        }
        logger.info(`已应用 ${appliedMerged} 个合并后的权重`);

        // 清理未使用的合并权重 (如果有的话)
        for (const tensor of mergedWeights.values()) {
            tensor.dispose();
        }

        // ==========================================
        // 步骤 4: 报告结果
        // ==========================================
        // 过滤掉：
        // 1. in_proj_weight/in_proj_bias - 已通过合并方式加载
        // 2. decoder.mask - 是预计算的 buffer，不需要从权重加载
        const actualMissing = result.missingKeys.filter(k =>
            !k.includes('.in_proj_weight') &&
            !k.includes('.in_proj_bias') &&
            k !== 'decoder.mask'
        );

        // 过滤掉已处理的 q/k/v_proj 相关的 unexpected
        const actualUnexpected = result.unexpectedKeys.filter(k =>
            !processedHFKeys.has(k)
        );

        const totalLoaded = result.loadedKeys.length + appliedMerged;
        logger.info(`已加载参数: ${totalLoaded}`);

        if (actualMissing.length > 0) {
            logger.warn(`缺失参数: ${actualMissing.length}`);
            for (const key of actualMissing.slice(0, 10)) {
                logger.warn(`  - ${key}`);
            }
            if (actualMissing.length > 10) {
                logger.warn(`  ... 还有 ${actualMissing.length - 10} 个`);
            }
        }

        if (actualUnexpected.length > 0) {
            logger.warn(`未预期参数: ${actualUnexpected.length}`);
            for (const key of actualUnexpected.slice(0, 10)) {
                logger.warn(`  - ${key}`);
            }
            if (actualUnexpected.length > 10) {
                logger.warn(`  ... 还有 ${actualUnexpected.length - 10} 个`);
            }
        }

        group.close();
    } catch (e) {
        logger.error(`加载模型权重失败: ${(e as Error).message}`);
        throw e;
    }
}

/**
 * 步骤 2: 加载模型
 * 创建 Whisper 模型实例并加载权重
 */
async function loadModel(): Promise<void> {
    logger.logGroup('步骤 2: 加载模型');

    try {
        const config = getWhisperConfig(MODEL_SIZE);
        logger.info(`创建 WhisperModel 实例 (${MODEL_SIZE})...`);
        logger.info(`  d_model: ${config.dModel}`);
        logger.info(`  encoder_layers: ${config.encoderLayers}`);
        logger.info(`  decoder_layers: ${config.decoderLayers}`);
        logger.info(`  n_mels: ${config.numMelBins}`);

        _model = new WhisperModel(config);

        const startTime = performance.now();
        await loadModelWeights(_model);
        const loadTime = performance.now() - startTime;
        logger.info(`权重加载耗时: ${loadTime.toFixed(0)}ms`);

        // 验证嵌入层权重的合法性 (HF 使用 embed_tokens)
        const embedWeight = _model.decoder.embed_tokens.weight;
        const embedStats = await getTensorStats(embedWeight);
        logger.info(
            `embed_tokens.weight: ` +
            `shape=${embedWeight.shape}, ` +
            `dtype=${embedWeight.dtype}, ` +
            `mean=${embedStats.mean.toFixed(6)}`
        );
        assert(embedStats.isFinite, 'embed_tokens.weight 必须包含有限值');

        logger.success('✓ 模型加载成功');
    } catch (e) {
        logger.error(`加载模型失败: ${(e as Error).message}`);
        throw e;
    }
}

function getModel(): WhisperModel {
    if (!_model) throw new Error('Model not loaded');
    return _model;
}

// ============================================================================
// 步骤 3: 预处理音频
// ============================================================================

/**
 * 步骤 3: 预处理音频
 * 加载 WAV 文件并计算 Mel 频谱图
 */
async function preprocessAudio(): Promise<Tensor> {
    logger.logGroup('步骤 3: 预处理音频');

    try {
        logger.info(`音频文件: ${AUDIO_FILE_PATH}`);

        // 创建特征提取器
        const config = getWhisperConfig(MODEL_SIZE);
        _featureExtractor = new WhisperFeatureExtractor(config.numMelBins);

        const startTime = performance.now();
        const melSpec = await _featureExtractor.fromFile(AUDIO_FILE_PATH);
        const processTime = performance.now() - startTime;

        logger.info(`Mel 频谱图形状: [${melSpec.shape.join(', ')}]`);
        logger.info(`预处理耗时: ${processTime.toFixed(0)}ms`);

        const melStats = await getTensorStats(melSpec);
        logger.info(
            `Mel 统计: min=${melStats.min.toFixed(4)}, max=${melStats.max.toFixed(4)}, mean=${melStats.mean.toFixed(4)}`
        );
        assert(melStats.isFinite, 'Mel 频谱图必须包含有限值');

        logger.success('✓ 音频预处理成功');
        return melSpec;
    } catch (e) {
        logger.error(`音频预处理失败: ${(e as Error).message}`);
        throw e;
    }
}

// ============================================================================
// 步骤 4: 推理演示
// ============================================================================

/**
 * 步骤 4: 运行推理演示
 * 使用 Whisper 模型进行语音转文本
 */
async function runInferenceDemo(melSpec: Tensor): Promise<void> {
    logger.logGroup('步骤 4: 语音转文本推理');

    try {
        const tokenizer = getTokenizer();
        const model = getModel();

        logger.info(`语言: ${LANGUAGE}`);
        logger.info(`任务: ${TASK}`);

        const startTime = performance.now();

        // 生成转录
        const generatedTokens = await model.generate(melSpec, {
            language: LANGUAGE,
            task: TASK,
            maxNewTokens: MAX_NEW_TOKENS,
            // maxNewTokens: 10,
            withTimestamps: false,
        });

        const totalTime = performance.now() - startTime;

        // 解码输出
        // 过滤掉特殊 token
        const filteredTokens = generatedTokens.filter(
            (t) => t < 50257  // 特殊 token 从 50257 开始
        );

        const transcription = tokenizer.decode(filteredTokens, {
            skip_special_tokens: true,
        });

        console.log('\n');
        logger.info('转录结果:');
        console.log(`  "${transcription}"`);
        console.log('\n');

        // 输出统计信息
        logger.info(`生成耗时: ${totalTime.toFixed(0)}ms`);
        logger.info(`生成 token 数: ${generatedTokens.length}`);
        logger.info(`生成速度: ${(generatedTokens.length / (totalTime / 1000)).toFixed(2)} tokens/s`);

        logger.success('✓ 演示完成');
    } catch (e) {
        logger.error(`推理演示失败: ${(e as Error).message}`);
        throw e;
    }
}

// ============================================================================
// 主函数
// ============================================================================

/**
 * 主函数：执行完整的 Whisper 推理流程
 */
async function main() {
    const startTime = performance.now();

    try {
        // 步骤 0: 初始化 WebGPU 后端
        logger.logGroup('步骤 0: 初始化 WebGPU 后端');
        await init();
        logger.success('✓ WebGPU 后端初始化成功');

        // 步骤 1-4: 加载和推理
        await loadTokenizer();
        await loadModel();
        const melSpec = await preprocessAudio();
        await runInferenceDemo(melSpec);

        // 清理
        melSpec.dispose();

        const totalTime = performance.now() - startTime;
        logger.success(`\n✅ 所有步骤完成！总耗时: ${(totalTime / 1000).toFixed(2)}s`);
    } catch (error) {
        logger.error(`\n❌ 执行失败: ${error instanceof Error ? error.message : String(error)}`);
        if (error instanceof Error && error.stack) {
            logger.error('\n堆栈跟踪:');
            logger.error(error.stack);
        }
        process.exit(1);
    }
}

// 执行主函数
main();
