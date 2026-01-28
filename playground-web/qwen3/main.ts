import { Tokenizer } from '@huggingface/tokenizers';
import { io, Tensor } from '@kandle/core';
import { Qwen3ForCausalLM } from '@kandle/model-utils';
import { QWEN3_CONFIG, logger, EOS_TOKEN_IDS } from './utils';
import { initWebGPU } from './init';

await initWebGPU();

// State
let _tokenizer: Tokenizer | null = null;
let _model: Qwen3ForCausalLM | null = null;

// UI Elements
const tokenizerInput = document.getElementById('tokenizerFile') as HTMLInputElement;
const modelInput = document.getElementById('modelFile') as HTMLInputElement;
const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement;
const generateBtn = document.getElementById('generateBtn') as HTMLButtonElement;
const promptInput = document.getElementById('promptInput') as HTMLTextAreaElement;
const tokenizerStatus = document.getElementById('tokenizerStatus') as HTMLSpanElement;
const modelStatus = document.getElementById('modelStatus') as HTMLSpanElement;

// Helper to update status
function checkReadyToLoad() {
    const hasTokenizer = tokenizerInput.files && tokenizerInput.files.length > 0;
    const hasModel = modelInput.files && modelInput.files.length > 0;
    loadBtn.disabled = !(hasTokenizer && hasModel);
}

// Event Listeners
tokenizerInput.addEventListener('change', () => {
    checkReadyToLoad();
    if (tokenizerInput.files?.[0]) {
        tokenizerStatus.textContent = `📄 ${tokenizerInput.files[0].name}`;
    }
});

modelInput.addEventListener('change', () => {
    checkReadyToLoad();
    if (modelInput.files?.[0]) {
        modelStatus.textContent = `📦 ${modelInput.files[0].name}`;
    }
});

loadBtn.addEventListener('click', async () => {
    loadBtn.disabled = true;
    try {
        await loadAll();
        generateBtn.disabled = false;
        alert('Model loaded successfully!');
    } catch (e) {
        logger.error(`Load failed: ${e}`);
        console.error(e);
        loadBtn.disabled = false;
    }
});

generateBtn.addEventListener('click', async () => {
    if (!_model || !_tokenizer) return;
    generateBtn.disabled = true;
    try {
        const prompt = promptInput.value;
        await generate(prompt);
    } catch (e) {
        logger.error(`Generation failed: ${e}`);
        console.error(e);
    } finally {
        generateBtn.disabled = false;
    }
});

async function loadAll() {
    logger.logGroup('Loading Model & Tokenizer');

    // 1. Load Tokenizer
    const tokenFile = tokenizerInput.files![0];
    const tokenText = await tokenFile.text();
    // Assuming tokenizer.json contains everything needed or we don't have config separately here
    // If config IS needed (tokenizer_config.json), we might need another input, but usually tokenizer.json is enough for basic decoding
    // However, the test code used both. Let's try init with just json first.
    // If user provides tokenizer.json it should be the full definition.

    // Note: HuggingFace Tokenizer.fromJSON takes the JSON object.
    const tokenizerJson = JSON.parse(tokenText);
    _tokenizer = new Tokenizer(tokenizerJson, {});
    // If we need 'model' and 'pre_tokenizer' etc which are usually in tokenizer.json.
    // Sometimes there is a tokenizer_config.json which has added tokens etc.
    // The test code loaded both. new Tokenizer(json, config)
    // For simplicity let's assume tokenizer.json is self-contained enough or we can try to improve if it fails.

    logger.success('✓ Tokenizer loaded');

    // 2. Load Model
    logger.info('Creating model instance...');
    _model = new Qwen3ForCausalLM(QWEN3_CONFIG, true);

    const modelFile = modelInput.files![0];
    logger.info(`Loading weights from ${modelFile.name} (${(modelFile.size / 1024 / 1024).toFixed(1)} MB)...`);

    // Use @kandle/core io to load safetensor
    // io.loadSafetensor accepts File object
    const group = await io.loadSafetensor(modelFile);
    logger.info(`Safetensor group loaded: ${group.layers.size} layers`);

    const result = await _model.loadFromSafetensor(group, {
        strict: false, // Allow missing keys (like rotary_emb.inv_freq often computed)
        keyMapper: (key) => key,
    });

    logger.info(`Loaded keys: ${result.loadedKeys.length}`);
    if (result.missingKeys.length > 0) {
        logger.warn(`Missing keys: ${result.missingKeys.length}`);
    }

    group.close();

    // Init LM Head
    _model.initLMHead();
    logger.success('✓ Model loaded');
}

async function generate(prompt: string) {
    logger.logGroup('Generation');
    logger.info(`Prompt: "${prompt}"`);

    if (!_tokenizer || !_model) throw new Error("Not initialized");

    const encoded = _tokenizer.encode(prompt);
    const inputIdsArray = [...encoded.ids];

    logger.info(`Input IDs: [${inputIdsArray.join(', ')}]`);

    const MAX_NEW_TOKENS = 50; // Limit for demo

    let currentInputIds = inputIdsArray;

    const startTime = performance.now();
    const generatedTokens: number[] = [];

    // 回调函数：每生成一个 token 就调用一次，实现流式输出
    const onTokenGenerated = (token: string, tokenId: number, isLast: boolean) => {
        logger.info(`+ "${token}"`);

        // 如果是最后一个 token，显示完整结果
        if (isLast) {
            const totalTime = performance.now() - startTime;
            logger.success(`Generation complete in ${totalTime.toFixed(0)}ms`);
            const fullResponse = _tokenizer!.decode(generatedTokens, { skip_special_tokens: true });
            logger.info(`Full Response: \n${fullResponse}`);
        }
    };

    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        // Create tensor
        const inputTensor = new Tensor(
            new Int32Array(currentInputIds),
            { dtype: 'int32', shape: [1, currentInputIds.length] }
        );

        const nextTokenId = await _model.generateNextToken(inputTensor);
        inputTensor.dispose();

        generatedTokens.push(nextTokenId);
        currentInputIds.push(nextTokenId);

        const decodedToken = _tokenizer.decode([nextTokenId]);

        // 检查是否结束
        const isEOS = EOS_TOKEN_IDS.includes(nextTokenId);
        const isLast = isEOS || i === MAX_NEW_TOKENS - 1;

        // 调用回调函数，实现流式输出
        onTokenGenerated(decodedToken, nextTokenId, isLast);

        if (isEOS) {
            logger.info('EOS generated');
            break;
        }
    }
}
