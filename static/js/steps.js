/**
 * Steps - 步骤处理模块
 * 处理各步骤的执行和状态显示
 */

// ============================================
// 全局状态
// ============================================

// 存储步骤2的推荐顺序
let step2Recommendations = [];

// ============================================
// 状态显示
// ============================================

/**
 * 显示状态消息
 * @param {HTMLElement} element - 消息容器
 * @param {string} type - 消息类型
 * @param {string} message - 消息内容
 */
function showStatus(element, type, message) {
    const icons = {
        success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
        error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };

    element.className = `status-message active status-${type}`;
    element.innerHTML = `${icons[type]} ${message}`;
}

/**
 * 显示任务监控器（含ETA倒计时）
 * @param {string} message - 状态消息
 * @param {number} progress - 进度百分比
 * @param {Object} timing - 可选：后端传来的计时数据 { elapsed_seconds, eta_seconds }
 * @param {boolean} isComplete - 是否已完成（显示总结而非ETA）
 * @param {string} timingHtml - 可选：完成后的计时HTML
 */
let _monitorTiming = { lastProgress: 0, lastTime: 0, startTime: 0 };

function _formatDuration(seconds) {
    if (!seconds || seconds < 0) return '--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function showTaskMonitor(message, progress, timing = null, isComplete = false, timingHtml = '') {
    const monitor = document.getElementById('task-monitor');
    const content = document.getElementById('task-monitor-content');

    if (!monitor || !content) return;

    monitor.classList.add('active');

    let etaHtml = '';
    let progressBarHtml = '';

    if (isComplete) {
        // 完成状态：显示总结
        progressBarHtml = `
            <div class="progress-bar" style="margin-top: 0.5rem;">
                <div class="progress-fill" style="width: 100%; background: linear-gradient(90deg, var(--success), #34d399);"></div>
            </div>
        `;
    } else {
        // 运行中：计算并显示ETA
        const now = Date.now();
        if (_monitorTiming.startTime === 0) {
            _monitorTiming.startTime = now;
            _monitorTiming.lastProgress = progress;
            _monitorTiming.lastTime = now;
        }

        // 优先使用后端传来的eta_seconds（更准确）
        let etaSec = null;
        if (timing && timing.eta_seconds != null) {
            etaSec = Math.max(0, Math.round(timing.eta_seconds));
        } else if (progress > 2 && progress < 99) {
            // 本地滚动平均估算
            const elapsed = (now - _monitorTiming.startTime) / 1000;
            const rate = progress / elapsed; // % per second
            if (rate > 0) {
                const remaining = (100 - progress) / rate;
                etaSec = Math.round(remaining);
            }
        }

        if (etaSec != null && etaSec >= 0) {
            const etaStr = _formatDuration(etaSec);
            etaHtml = `<div class="task-monitor-eta">预计剩余: ${etaStr}</div>`;
        }

        progressBarHtml = `
            <div class="progress-bar" style="margin-top: 0.5rem;">
                <div class="progress-fill" style="width: ${progress}%;"></div>
            </div>
        `;
    }

    content.innerHTML = `
        <div class="task-monitor-status">${message}</div>
        ${progressBarHtml}
        ${etaHtml}
        ${timingHtml ? `<div class="task-monitor-timing">${timingHtml}</div>` : ''}
    `;
}

/**
 * 重置任务监控器计时状态（新任务开始时调用）
 */
function resetTaskMonitorTiming() {
    _monitorTiming = { lastProgress: 0, lastTime: 0, startTime: 0 };
}

/**
 * 隐藏任务监控器
 */
function hideTaskMonitor() {
    setTimeout(() => {
        const monitor = document.getElementById('task-monitor');
        if (monitor) {
            monitor.classList.remove('active');
        }
    }, 3000);
}

/**
 * 更新进度显示
 * @param {number} stepNum - 步骤编号
 * @param {Object} data - 状态数据
 */
function updateProgress(stepNum, data) {
    const progressFill = document.getElementById(`step${stepNum}-progress-fill`);
    const progressText = document.getElementById(`step${stepNum}-status`);
    const progressPercent = document.getElementById(`step${stepNum}-percent`);

    if (progressFill) {
        progressFill.style.width = `${data.progress}%`;
    }
    if (progressText) {
        progressText.textContent = data.message;
    }
    if (progressPercent) {
        progressPercent.textContent = `${Math.round(data.progress)}%`;
    }

    showTaskMonitor(`步骤${stepNum}: ${data.message}`, data.progress);
}

// ============================================
// 步骤完成处理
// ============================================

/**
 * 处理步骤完成
 * @param {number} stepNum - 步骤编号
 * @param {Object} result - 结果数据
 */
function handleStepComplete(stepNum, result) {
    const messageBox = document.getElementById(`step${stepNum}-message`);
    const progressFill = document.getElementById(`step${stepNum}-progress-fill`);
    const progressText = document.getElementById(`step${stepNum}-status`);
    const progressPercent = document.getElementById(`step${stepNum}-percent`);

    if (stepNum === 2 && result && result.recommendations) {
        step2Recommendations = result.recommendations;

        const tbody = document.getElementById('step2-recommendations');
        if (tbody) {
            tbody.innerHTML = result.recommendations.map(r => `
                <tr>
                    <td><span class="match-order-item">${r.rank}</span></td>
                    <td>${r.class_name}</td>
                    <td>${r.confusion_score}</td>
                </tr>
            `).join('');
        }

        const resultsDiv = document.getElementById('step2-results');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }

        // 显示提示卡片
        const preview = result.recommendations.slice(0, 3).map(r => `${r.rank}.${r.class_name}`).join(' > ');
        const tipPreview = document.getElementById('step2-tip-preview');
        const tipCard = document.getElementById('step2-tip-card');

        if (tipPreview) {
            tipPreview.textContent = preview + (result.recommendations.length > 3 ? ' ...' : '');
        }
        if (tipCard) {
            tipCard.style.display = 'block';
        }
    }

    if (stepNum === 6 && result) {
        const oa = document.getElementById('step6-oa');
        const kappa = document.getElementById('step6-kappa');
        const pixels = document.getElementById('step6-pixels');
        const tbody = document.getElementById('step6-class-accuracies');
        const resultsDiv = document.getElementById('step6-results');

        if (oa) oa.textContent = result.metrics.overall_accuracy + '%';
        if (kappa) kappa.textContent = result.metrics.kappa;
        if (pixels) pixels.textContent = result.metrics.total_pixels;

        if (tbody && result.class_accuracies) {
            tbody.innerHTML = result.class_accuracies.map(c => `
                <tr>
                    <td>${c.class_name}</td>
                    <td>${c.producer_accuracy}%</td>
                    <td>${c.user_accuracy}%</td>
                </tr>
            `).join('');
        }

        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
    }

    // 显示成功消息
    if (messageBox) {
        showStatus(messageBox, 'success', `处理完成！${result?.message || ''}`);
    }

    // 更新进度条
    if (progressFill) progressFill.style.width = '100%';
    if (progressText) progressText.textContent = '处理完成';
    if (progressPercent) progressPercent.textContent = '100%';

    hideTaskMonitor();
}

/**
 * 处理步骤错误
 * @param {number} stepNum - 步骤编号
 * @param {Object} data - 错误数据
 */
function handleStepError(stepNum, data) {
    const messageBox = document.getElementById(`step${stepNum}-message`);
    const progressText = document.getElementById(`step${stepNum}-status`);

    if (messageBox) {
        showStatus(messageBox, 'error', `错误: ${data.error || '未知错误'}`);
    }
    if (progressText) {
        progressText.textContent = '处理失败';
    }

    hideTaskMonitor();
}

// ============================================
// 步骤执行
// ============================================

/**
 * 运行步骤1
 */
/**
 * 消融实验1：步骤1「跳过离散化」开关联动
 * 勾选后自动勾上逐波段标准化（连续数值下反射率波段量级会压倒指数波段）
 */
function onStep1SkipDiscretizationChange(el) {
    const norm = document.getElementById('step1-normalize-continuous');
    if (el.checked && norm && !norm.checked) {
        norm.checked = true;
    }
}

// ============================================
// 零值开关（含 0 值 / 排除 0 值）
// ============================================

const ZERO_SWITCH_HINTS = {
    on:  '<b>开关打开 = 含 0 值</b>：0 一并计入分位数统计，等级严格等分，背景自然独占等级 1 —— 与 92.66% 历史高精度实验口径一致。实测（124036 全时相 2 级离散）：评价样本 98.17%；实验样本_7月版 含0 = 95.71% vs 排除0 = 89.91%（<b>+5.80pp</b>）。注意：指数波段（NDBI/NDWI 等）的 0 与负值是合法观测值而非背景，此时开关应保持打开。',
    off: '<b>开关关闭 = 排除 0 值</b>：分位数统计只取 &gt;0 的值，0 与负值均不参与，分位点只反映有效地物、有效地物独占完整等级分辨率。实测（124036 全时相 2 级离散）：实验样本_7月版 89.91%（低于含 0 值的 95.71%）；评价样本 98.17%。仅在 0/负值确实代表背景、且有效地物量级偏小时考虑关闭。',
    auto:'<b>自动择优</b>：按采样统计的 0 值占比与等级占用情况自动判定，判定依据会写入 step1/merged.csv.meta.json。'
};

/**
 * 零值开关切换（含 0 ↔ 排除 0）
 * @param {string} scope - 'step1' | 'pipeline'
 */
function onZeroSwitchChange(scope) {
    const sw   = document.getElementById(scope + '-zero-switch');
    const auto = document.getElementById(scope + '-zero-auto');
    if (!sw) return;
    // 手动拨动开关时，取消「自动择优」
    if (auto && auto.checked) auto.checked = false;
    syncZeroSwitchUI(scope);
}

/**
 * 「自动择优」勾选联动
 * @param {string} scope - 'step1' | 'pipeline'
 */
function onZeroAutoChange(scope) {
    const sw   = document.getElementById(scope + '-zero-switch');
    const auto = document.getElementById(scope + '-zero-auto');
    if (!sw || !auto) return;
    // 自动模式下开关置灰，由后端决定
    sw.disabled = auto.checked;
    syncZeroSwitchUI(scope);
}

/**
 * 同步零值开关的文案与提示
 * @param {string} scope - 'step1' | 'pipeline'
 */
function syncZeroSwitchUI(scope) {
    const sw   = document.getElementById(scope + '-zero-switch');
    const auto = document.getElementById(scope + '-zero-auto');
    const txt  = document.getElementById(scope + '-zero-switch-text');
    const hint = document.getElementById(scope + '-zero-hint');
    if (!sw || !txt || !hint) return;

    const isAuto = auto && auto.checked;
    if (isAuto) {
        txt.textContent = '自动择优（后端判定）';
        txt.className = 'switch-text auto';
        hint.innerHTML = ZERO_SWITCH_HINTS.auto;
    } else {
        const on = sw.checked;
        txt.textContent = on ? '含 0 值（0 参与分位数统计）'
                             : '排除 0 值（只取 >0 的值）';
        txt.className = 'switch-text' + (on ? '' : ' off');
        hint.innerHTML = on ? ZERO_SWITCH_HINTS.on : ZERO_SWITCH_HINTS.off;
    }
}

/**
 * 读取零值开关当前值，返回后端 zero_policy 枚举
 * @param {string} scope - 'step1' | 'pipeline'
 * @returns {string} 'auto' | 'include' | 'exclude'
 */
function getZeroPolicy(scope) {
    const sw   = document.getElementById(scope + '-zero-switch');
    const auto = document.getElementById(scope + '-zero-auto');
    if (auto && auto.checked) return 'auto';
    if (!sw) return 'include';
    return sw.checked ? 'include' : 'exclude';
}

// 页面加载后初始化两个面板的开关状态
document.addEventListener('DOMContentLoaded', function () {
    ['step1', 'pipeline'].forEach(function (scope) {
        const auto = document.getElementById(scope + '-zero-auto');
        const sw   = document.getElementById(scope + '-zero-switch');
        if (auto && sw) sw.disabled = auto.checked;
        syncZeroSwitchUI(scope);
    });
});

async function runStep1() {
    const input = document.getElementById('step1-input-img-dir');
    const output = document.getElementById('step1-output-dir');
    const filename = document.getElementById('step1-filename');
    const numClasses = document.getElementById('step1-num-classes');

    if (!input.value || !output.value || !filename.value) {
        alert('请填写所有必填字段');
        return;
    }

    // 获取删除波段模式
    const deleteModeEl = document.querySelector('#step1-delete-mode .radio-option.active');
    const deleteMode = deleteModeEl ? deleteModeEl.dataset.value : 'default';

    let deleteRanges = null;
    if (deleteMode === 'custom') {
        deleteRanges = Array.from(document.querySelectorAll('#step1-delete-ranges .dynamic-item'))
            .map(item => {
                const inputs = item.querySelectorAll('input');
                return { start: parseInt(inputs[0].value), end: parseInt(inputs[1].value) };
            })
            .filter(r => r.start && r.end);
        if (deleteRanges.length === 0) deleteRanges = null;
    }

    const progress = document.getElementById('step1-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep1({
            inputImgDir: input.value,
            outputDir: output.value,
            filename: filename.value,
            numClasses: parseInt(numClasses.value),
            deleteRanges: deleteRanges,
            adjustSizes: document.getElementById('step1-adjust-sizes').checked,
            skipDiscretization: !!(document.getElementById('step1-skip-discretization') && document.getElementById('step1-skip-discretization').checked),  // 消融实验1
            normalizeContinuous: !!(document.getElementById('step1-normalize-continuous') && document.getElementById('step1-normalize-continuous').checked),  // 消融实验1配套：逐波段标准化
            zeroPolicy:           getZeroPolicy('step1')  // 零值开关：含0=include / 排除0=exclude / 自动=auto
        });

        // 开始轮询状态
        TaskAPI.pollStatus(
            result.task_id,
            1,
            (data) => updateProgress(1, data),
            (data) => handleStepComplete(1, data.result),
            (data) => handleStepError(1, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step1-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}

/**
 * 运行步骤2
 */
async function runStep2() {
    const shpInput = document.getElementById('step2-shp-input');
    const tifFolder = document.getElementById('step2-tif-folder');
    const experimentData = document.getElementById('step2-experiment-data');
    const outputDir = document.getElementById('step2-output-dir');

    if (!shpInput.value || !tifFolder.value || !experimentData.value || !outputDir.value) {
        alert('请填写所有必填字段');
        return;
    }

    const progress = document.getElementById('step2-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep2({
            shpInput: shpInput.value,
            tifFolder: tifFolder.value,
            experimentData: experimentData.value,
            outputDir: outputDir.value
        });

        TaskAPI.pollStatus(
            result.task_id,
            2,
            (data) => updateProgress(2, data),
            (data) => handleStepComplete(2, data.result),
            (data) => handleStepError(2, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step2-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}

/**
 * 加载步骤2推荐顺序
 */
function loadStep2Recommendations() {
    if (step2Recommendations.length === 0) {
        alert('请先运行步骤2获取推荐顺序');
        return;
    }

    const container = document.getElementById('step3-match-order');
    if (container) {
        container.innerHTML = step2Recommendations.map(r =>
            `<span class="match-order-item">${r.rank}. ${r.class_name}</span>`
        ).join('');
    }

    // 同时填充轮次配置
    const roundsContainer = document.getElementById('step3-rounds');
    if (roundsContainer && roundsContainer.querySelectorAll('.dynamic-item').length === 0) {
        addRound();
    }

    const replacements = document.querySelector('#step3-rounds .round-replacements');
    if (replacements) {
        step2Recommendations.forEach(r => {
            addReplacement(replacements.previousElementSibling);
            const items = replacements.parentElement.querySelectorAll('.dynamic-item');
            const lastItem = items[items.length - 1];
            if (lastItem) {
                const sampleInput = lastItem.querySelector('.replacement-sample');
                const valueInput = lastItem.querySelector('.replacement-value');
                if (sampleInput) sampleInput.value = r.class_name + '.csv';
                if (valueInput) valueInput.value = r.rank;
            }
        });
    }
}

/**
 * 运行步骤3
 */
async function runStep3() {
    const imagesFolder = document.getElementById('step3-images-folder');
    const samplesFolder = document.getElementById('step3-samples-folder');
    const targetFile = document.getElementById('step3-target-file');
    const outputFile = document.getElementById('step3-output-file');

    if (!imagesFolder.value || !samplesFolder.value || !targetFile.value || !outputFile.value) {
        alert('请填写所有必填字段');
        return;
    }

    // 获取匹配模式
    const matchModeEl = document.querySelector('#step3-match-mode .radio-option.active');
    const matchMode = matchModeEl ? matchModeEl.dataset.value : 'default';

    // 获取距离度量（消融实验2：汉明距离 / 欧氏距离）
    const distanceMetricEl = document.querySelector('#step3-distance-metric .radio-option.active');
    const distanceMetric = distanceMetricEl ? distanceMetricEl.dataset.value : 'hamming';

    // 构建匹配顺序
    let matchOrder = [];
    if (matchMode === 'default') {
        const orderContainer = document.getElementById('step3-match-order');
        if (orderContainer) {
            matchOrder = Array.from(orderContainer.querySelectorAll('.match-order-item'))
                .map(el => el.textContent.split('. ')[1] + '.csv');
        }
    } else {
        const customContainer = document.getElementById('step3-custom-match-list');
        if (customContainer) {
            matchOrder = Array.from(customContainer.querySelectorAll('.dynamic-item input'))
                .map(input => input.value.trim())
                .filter(v => v.length > 0)
                .map(name => name + '.csv');
        }
    }

    if (matchOrder.length === 0) {
        alert('请先配置匹配顺序');
        return;
    }

    // 构建轮次配置
    const rounds = Array.from(document.querySelectorAll('#step3-rounds .dynamic-item')).map(roundDiv => {
        const threshold = roundDiv.querySelector('.round-threshold').value;
        const replacements = {};
        roundDiv.querySelectorAll('.dynamic-item').forEach(item => {
            const sample = item.querySelector('.replacement-sample').value;
            const value = item.querySelector('.replacement-value').value;
            if (sample && value) {
                replacements[sample] = value;
            }
        });
        return { threshold: parseFloat(threshold), replacements };
    });

    const progress = document.getElementById('step3-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep3({
            imagesFolder: imagesFolder.value,
            samplesFolder: samplesFolder.value,
            targetFile: targetFile.value,
            outputFile: outputFile.value,
            matchOrder: matchOrder,
            rounds: rounds,
            distanceMetric: distanceMetric,  // 消融实验2：距离度量选择
            hammingTolerance: parseFloat(document.getElementById('step3-hamming-tolerance')?.value || '0') || 0  // 消融实验1配套：汉明容差 ε
        });

        TaskAPI.pollStatus(
            result.task_id,
            3,
            (data) => updateProgress(3, data),
            (data) => handleStepComplete(3, data.result),
            (data) => handleStepError(3, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step3-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}

/**
 * 运行步骤4
 */
async function runStep4() {
    const inputCsv = document.getElementById('step4-input-csv');
    const imagesFolder = document.getElementById('step4-images-folder');
    const outputDir = document.getElementById('step4-output-dir');
    const maxSeqLength = document.getElementById('step4-max-seq-length');

    if (!inputCsv.value || !imagesFolder.value || !outputDir.value) {
        alert('请填写所有必填字段');
        return;
    }

    const progress = document.getElementById('step4-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep4({
            inputCsv: inputCsv.value,
            imagesFolder: imagesFolder.value,
            outputDir: outputDir.value,
            addGeo: document.getElementById('step4-add-geo').checked,
            maxSeqLength: maxSeqLength.value ? parseInt(maxSeqLength.value) : null
        });

        TaskAPI.pollStatus(
            result.task_id,
            4,
            (data) => updateProgress(4, data),
            (data) => handleStepComplete(4, data.result),
            (data) => handleStepError(4, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step4-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}

/**
 * 运行步骤5
 */
async function runStep5() {
    const datPath = document.getElementById('step5-dat-path');

    if (!datPath.value) {
        alert('请选择DAT文件');
        return;
    }

    const classes = Array.from(document.querySelectorAll('#step5-classes .dynamic-item')).map(item => ({
        name: item.querySelector('.class-name').value || '未命名',
        values: item.querySelector('.class-values').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
    })).filter(c => c.values.length > 0);

    if (classes.length === 0) {
        alert('请至少添加一个地物类别');
        return;
    }

    const progress = document.getElementById('step5-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep5({
            datPath: datPath.value,
            classes: classes
        });

        TaskAPI.pollStatus(
            result.task_id,
            5,
            (data) => updateProgress(5, data),
            (data) => handleStepComplete(5, data.result),
            (data) => handleStepError(5, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step5-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}

/**
 * 运行步骤6
 */
async function runStep6() {
    const reclassDat = document.getElementById('step6-reclass-dat');
    const shpFolder = document.getElementById('step6-shp-folder');
    const outputDir = document.getElementById('step6-output-dir');

    if (!reclassDat.value || !shpFolder.value || !outputDir.value) {
        alert('请填写所有必填字段');
        return;
    }

    const progress = document.getElementById('step6-progress');
    if (progress) progress.classList.add('active');

    try {
        const result = await StepsAPI.runStep6({
            reclassDat: reclassDat.value,
            shpFolder: shpFolder.value,
            outputDir: outputDir.value
        });

        TaskAPI.pollStatus(
            result.task_id,
            6,
            (data) => updateProgress(6, data),
            (data) => handleStepComplete(6, data.result),
            (data) => handleStepError(6, data)
        );
    } catch (error) {
        const messageBox = document.getElementById('step6-message');
        if (messageBox) {
            showStatus(messageBox, 'error', `请求失败: ${error.message}`);
        }
    }
}
