/**
 * Qwen3 端到端推理示例 - Node.js 版本
 * 
 * 使用方法：npm run qwen3
 * 
 * 工作流程：
 * 1. 初始化 WebGPU 后端
 * 2. 加载 tokenizer
 * 3. 加载 Qwen3 模型权重
 * 4. 运行推理演示（ChatML 格式对话）
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { Tensor, io } from '@kandle/core';
import { Qwen3ForCausalLM } from '@kandle/model-utils';
import { Tokenizer } from '@huggingface/tokenizers';
import { init } from '../init';
import {
    QWEN3_CONFIG,
    EOS_TOKEN_IDS,
    logger,
    getTensorStats,
    assert,
} from './utils';

// ============================================================================
// 配置常量 - 在此修改模型路径和推理参数
// ============================================================================

/** 模型根目录（修改为你的模型路径） */
const MODEL_ROOT_DIR = '/models/qwen3-0.6b';

/** 派生路径 */
const MODEL_PATH = path.join(MODEL_ROOT_DIR, 'model.safetensors');
const TOKENIZER_JSON_PATH = path.join(MODEL_ROOT_DIR, 'tokenizer.json');
const TOKENIZER_CONFIG_PATH = path.join(MODEL_ROOT_DIR, 'tokenizer_config.json');

/** 最大生成 token 数量 */
const MAX_NEW_TOKENS = 512;

// ============================================================================
// 状态管理
// ============================================================================

let _tokenizer: Tokenizer | null = null;
let _model: Qwen3ForCausalLM | null = null;

// ============================================================================
// 步骤 1: 加载 Tokenizer
// ============================================================================

/**
 * 步骤 1: 加载 Tokenizer
 * 从本地文件系统读取 tokenizer.json 和 tokenizer_config.json
 */
async function loadTokenizer(): Promise<void> {
    logger.logGroup('步骤 1: 加载 Tokenizer');

    try {
        logger.info(`从目录加载: ${MODEL_ROOT_DIR}`);

        // 检查文件是否存在
        await Promise.all([
            fs.access(TOKENIZER_JSON_PATH),
            fs.access(TOKENIZER_CONFIG_PATH),
        ]).catch(() => {
            throw new Error(
                `Tokenizer 文件不存在。请检查路径配置：\n` +
                `  MODEL_ROOT_DIR = ${MODEL_ROOT_DIR}`
            );
        });

        // 读取并解析 tokenizer 文件
        const [jsonContent, configContent] = await Promise.all([
            fs.readFile(TOKENIZER_JSON_PATH, 'utf-8'),
            fs.readFile(TOKENIZER_CONFIG_PATH, 'utf-8'),
        ]);

        const tokenizerJson = JSON.parse(jsonContent);
        const tokenizerConfig = JSON.parse(configContent);

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
 */
async function loadModelWeights(model: Qwen3ForCausalLM): Promise<void> {
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

        // 加载到模型
        const group = await io.loadSafetensor(arrayBuffer);
        logger.info(`SafeTensor 加载: ${group.layers.size} 层`);

        const result = await model.loadFromSafetensor(group, {
            strict: false,
            keyMapper: (key) => key,
        });

        logger.info(`已加载参数: ${result.loadedKeys.length}`);
        if (result.missingKeys.length > 0) {
            logger.warn(`缺失参数: ${result.missingKeys.length}`);
        }
        if (result.unexpectedKeys.length > 0) {
            logger.warn(`未预期参数: ${result.unexpectedKeys.length}`);
        }

        group.close();
    } catch (e) {
        logger.error(`加载模型权重失败: ${(e as Error).message}`);
        throw e;
    }
}

/**
 * 步骤 2: 加载模型
 * 创建 Qwen3 模型实例并加载权重
 */
async function loadModel(): Promise<void> {
    logger.logGroup('步骤 2: 加载模型');

    try {
        logger.info('创建 Qwen3ForCausalLM 模型实例...');
        _model = new Qwen3ForCausalLM(QWEN3_CONFIG, true);

        const startTime = performance.now();
        await loadModelWeights(_model);
        const loadTime = performance.now() - startTime;
        logger.info(`权重加载耗时: ${loadTime.toFixed(0)}ms`);

        // 初始化语言模型头
        _model.initLMHead();

        // 验证嵌入层权重的合法性
        const embedWeight = _model.model.embed_tokens.weight;
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

function getModel(): Qwen3ForCausalLM {
    if (!_model) throw new Error('Model not loaded');
    return _model;
}

// ============================================================================
// 步骤 3: 推理演示
// ============================================================================

/**
 * 步骤 3: 运行推理演示
 * 使用 ChatML 格式进行对话生成
 */
async function runInferenceDemo(): Promise<void> {
    logger.logGroup('步骤 3: 对话生成演示');
    
    try {
        const tokenizer = getTokenizer();
        const model = getModel();

        // ChatML 格式的对话提示词
        const prompt = `<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
`;

        logger.info('输入提示词 (ChatML 格式):');
        logger.info(prompt);

        // 编码提示词
        const encoded = tokenizer.encode(prompt);
        logger.info(`提示词 token 数量: ${encoded.ids.length}`);

        // 自回归生成
        const tokenIds = [...encoded.ids];
        const startTime = performance.now();
        const generatedTokens: number[] = [];

        process.stdout.write('生成中: ');

        for (let i = 0; i < MAX_NEW_TOKENS; i++) {
            // 创建输入张量
            const inputIds = new Tensor(
                new Int32Array(tokenIds),
                { dtype: 'int32', shape: [1, tokenIds.length] }
            );

            // 生成下一个 token
            const nextTokenId = await model.generateNextToken(inputIds);
            inputIds.dispose();

            // 解码并显示
            const nextToken = tokenizer.decode([nextTokenId]);
            process.stdout.write(nextToken);

            generatedTokens.push(nextTokenId);

            // 检查是否遇到结束符
            if (EOS_TOKEN_IDS.includes(nextTokenId)) {
                logger.info(`\n[检测到 EOS token，生成结束]`);
                break;
            }

            tokenIds.push(nextTokenId);
        }

        const totalTime = performance.now() - startTime;
        console.log('\n');

        // 输出统计信息
        const response = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
        logger.info(`完整回复（清理后）: "${response}"`);
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
 * 主函数：执行完整的 Qwen3 推理流程
 */
async function main() {
    const startTime = performance.now();
    
    try {
        // 步骤 0: 初始化 WebGPU 后端
        logger.logGroup('步骤 0: 初始化 WebGPU 后端');
        await init();
        logger.success('✓ WebGPU 后端初始化成功');

        // 步骤 1-3: 加载和推理
        await loadTokenizer();
        await loadModel();
        await runInferenceDemo();

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

// 仅在直接执行时运行（非 import 时）
if (require.main === module) {
    main();
}

export { main };
