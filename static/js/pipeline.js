/**
 * Pipeline - 全流程处理模块（多轮迭代版）
 * 步骤1-2执行一次，步骤3-6循环迭代
 * 终止条件：Kappa≥阈值 或 未分类像素比例≥阈值
 */

/**
 * 加载替换值预览
 * 根据输入参数获取自动生成的替换值预览
 */
async function loadReplacementPreview() {
    const previewSection = document.getElementById('pipeline-replacement-preview-section');
    const previewContainer = document.getElementById('pipeline-replacement-preview');
    const classesCount = document.getElementById('pipeline-classes-count');

    if (!previewSection || !previewContainer) return;

    // 使用基础路径中的影像文件夹，无需单独输入
    const tifFolder = document.getElementById('pipeline-input-img-dir')?.value?.trim();
    const shpInput = document.getElementById('pipeline-shp-input')?.value?.trim();

    if (!tifFolder || !shpInput) {
        previewSection.style.display = 'none';
        return;
    }

    // 构造步骤1输出路径（若已存在则传给预览API获取真实匹配顺序）
    const outputDir = document.getElementById('pipeline-output-dir')?.value?.trim() || '';
    const filename  = document.getElementById('pipeline-filename')?.value?.trim() || 'merged';
    const experimentDataPath = outputDir ? `${outputDir}/step1/${filename}.csv` : null;

    // ★ 显示加载状态
    const placeholder = document.getElementById('pipeline-classes-placeholder');
    if (placeholder) {
        placeholder.innerHTML = `<span style="color:var(--text-muted);">⏳ 正在扫描 SHP 文件夹检测地物类别...</span>`;
        placeholder.style.padding = '1rem';
    }
    previewSection.style.display = 'block';
    if (previewContainer) previewContainer.innerHTML = '<p style="color: var(--text-muted);">正在连接服务器...</p>';

    try {
        // 调用后端API获取预览（使用统一API模块，较短超时）
        const data = await API.post('/pipeline/preview-replacements', {
            input_img_dir: tifFolder,
            shp_input: shpInput,
            num_rounds: parseInt(document.getElementById('pipeline-max-iterations')?.value) || 5,
            experiment_data_path: experimentDataPath,
        });

        pipelinePreviewMatchOrder = [];
        pipelinePreviewRounds = [];

        if (data.error) {
            previewSection.style.display = 'none';
            // 在类别占位区域显示具体错误原因
            if (placeholder) {
                placeholder.innerHTML = `<span style="color:#ef4444;">⚠ 自动检测失败: ${data.error}</span><br><span style="color:var(--text-muted);font-size:0.85rem;">请检查路径和文件后重新填写</span>`;
                placeholder.style.padding = '1rem';
            }
            console.warn('[Preview] API返回错误:', data.error);
            return;
        }

        pipelinePreviewMatchOrder = Array.isArray(data.match_order) ? data.match_order : [];
        pipelinePreviewRounds = Array.isArray(data.previews) ? data.previews : [];

        // ★ 优先使用 detected_classes（严格基于 .shp 文件），确保与后端一致
        const detectedClasses = data.detected_classes || [];
        if (detectedClasses.length > 0 && pipelinePreviewMatchOrder.length === 0) {
            pipelinePreviewMatchOrder = detectedClasses.map(n => `${n}.csv`);
            // 为检测到的类别生成默认替换值
            pipelinePreviewRounds = (data.num_classes || detectedClasses.length) > 0 ? [{
                round: 1,
                replacements: Object.fromEntries(
                    detectedClasses.map((name, i) => [`${name}.csv`, String(i + 1)])
                )
            }] : [];
        }
        renderReplacementPreview(data.previews, data.num_classes);
        syncPipelineClassesFromPreview();

        if (classesCount) {
            classesCount.textContent = `（检测到 ${data.num_classes} 个类别，共 ${data.sequence_length} 个时相）`;
        }

        previewSection.style.display = 'block';
    } catch (error) {
        console.error('加载替换值预览失败:', error);
        previewSection.style.display = 'none';
        // 显示网络错误或异常到界面
        if (placeholder) {
            const errMsg = error.message || String(error);
            placeholder.innerHTML = `<span style="color:#ef4444;">⚠ 检测请求失败: ${errMsg}</span><br><span style="color:var(--text-muted);font-size:0.85rem;">请确认服务器正在运行</span>`;
            placeholder.style.padding = '1rem';
        }
    }
}

/**
 * 渲染替换值预览表格
 */
function renderReplacementPreview(previews, numClasses) {
    const container = document.getElementById('pipeline-replacement-preview');
    if (!container) return;

    if (!previews || previews.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">无法生成预览，请检查输入参数</p>';
        return;
    }

    // 提取所有类别名称（从第一轮的replacements的key中获取）
    const firstRound = previews[0];
    if (!firstRound || !firstRound.replacements) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">无匹配样本</p>';
        return;
    }

    // 优先按 match_order 顺序排列列，确保与实际匹配顺序一致
    const allKeys = Object.keys(firstRound.replacements);
    const classNames = pipelinePreviewMatchOrder.length > 0
        ? pipelinePreviewMatchOrder.filter(k => allKeys.includes(k))
        : allKeys;

    // 生成表格HTML
    let html = `<table style="width:100%; font-size: 0.85rem; border-collapse: collapse;">
        <thead>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <th style="text-align: left; padding: 0.25rem 0.5rem; color: var(--text-muted);">轮次</th>
                ${classNames.map(name => `<th style="text-align: center; padding: 0.25rem 0.5rem; color: var(--text-muted);">${name}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${previews.slice(0, 5).map(round => `
                <tr style="border-bottom: 1px solid var(--border-color);">
                    <td style="padding: 0.25rem 0.5rem; font-weight: 500;">第 ${round.round} 轮</td>
                    ${classNames.map(name => `
                        <td style="text-align: center; padding: 0.25rem 0.5rem;">
                            <span class="replacement-badge">${round.replacements[name] || '-'}</span>
                        </td>
                    `).join('')}
                </tr>
            `).join('')}
        </tbody>
    </table>`;

    if (previews.length > 5) {
        html += `<p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem;">... 共 ${previews.length} 轮（实际运行中按需生成）</p>`;
    }

    container.innerHTML = html;
}

function syncPipelineClassesFromPreview() {
    const container = document.getElementById('pipeline-classes');
    if (!container || pipelinePreviewMatchOrder.length === 0 || pipelinePreviewRounds.length === 0) {
        return;
    }

    container.innerHTML = '';
    pipelinePreviewMatchOrder.forEach(sampleName => {
        const className = sampleName.replace(/\.csv$/i, '');
        const values = pipelinePreviewRounds
            .map(round => round?.replacements?.[sampleName])
            .filter(value => value !== undefined && value !== null && String(value).trim() !== '')
            .map(value => String(value))
            .join(',');

        addPipelineClass(className, values);
    });
}

// ============================================
// 全流程状态
// ============================================

const PIPELINE_STEPS = [
    { id: 1, name: '影像填充合并', icon: '01' },
    { id: 2, name: '样本进阶分析', icon: '02' },
    { id: 3, name: '汉明距离匹配', icon: '03' },
    { id: 4, name: 'ENVI格式转换', icon: '04' },
    { id: 5, name: '重分类优化',   icon: '05' },
    { id: 6, name: '精度评价',     icon: '06' },
];

let pipelineRunning = false;
let pipelinePaused = false;
let pipelineResuming = false;  // 刚恢复/重开标志，防止第一轮仍见paused就停
let currentTaskId = null;
let pipelinePreviewMatchOrder = [];
let pipelinePreviewRounds = [];

// 心跳计时器ID（浏览器关闭自动停止）
let _heartbeatIntervalId = null;

// ============================================
// 浏览器关闭自动停止
// ============================================

// beforeunload：窗口关闭时立即发送停止信号（sendBeacon 保证发送完成）
window.addEventListener('beforeunload', () => {
    if (currentTaskId && pipelineRunning) {
        const url = `${AppConfig.api.baseUrl}/tasks/${currentTaskId}/auto-stop`;
        navigator.sendBeacon(url, JSON.stringify({}));
    }
});

function _startHeartbeat(taskId) {
    _stopHeartbeat();  // 确保没有重复的
    // 每 10 秒发一次心跳，后端 30s 无心跳自动暂停
    _heartbeatIntervalId = setInterval(() => {
        if (!currentTaskId) {
            _stopHeartbeat();
            return;
        }
        // fire-and-forget，不阻塞轮询
        fetch(`${AppConfig.api.baseUrl}/tasks/${taskId}/heartbeat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        }).catch(() => {});  // 静默忽略错误
    }, 10000);
}

function _stopHeartbeat() {
    if (_heartbeatIntervalId) {
        clearInterval(_heartbeatIntervalId);
        _heartbeatIntervalId = null;
    }
}

// 日志收集（每条 {time, level, message, duration}）
let pipelineLogs = [];

// 步骤计时（步骤开始时间戳，键为 stepId）
let _stepStartTimes = {};
let _lastPolledStep = 0;

// Dashboard 图表实例
let dashAccuracyChart = null;
let dashSpectralChart = null;
let dashProcessSteps = [];
let dashRoundHistory = [];
let dashSpectralData = null;  // 步骤2返回的光谱指纹数据 {className: [band1, band2, ...]}

// ============================================
// 步骤状态面板
// ============================================

function initPipelineSteps() {
    PIPELINE_STEPS.forEach(s => {
        const el = document.getElementById(`pipeline-step-${s.id}`);
        if (!el) return;
        el.className = 'pipeline-step-card';
        const statusEl   = el.querySelector('.pipeline-step-status');
        const progressBar = el.querySelector('.pipeline-step-progress-fill');
        const msgEl      = el.querySelector('.pipeline-step-msg');
        if (statusEl)   statusEl.textContent    = '等待中';
        if (progressBar) progressBar.style.width = '0%';
        if (msgEl)      msgEl.textContent        = '';
    });
    // 清空轮次历史
    _clearRoundsHistory();
    // 隐藏终止原因横幅
    const banner = document.getElementById('pipeline-stop-banner');
    if (banner) banner.style.display = 'none';
}

function updatePipelineStep(stepId, status, msg, pct) {
    const el = document.getElementById(`pipeline-step-${stepId}`);
    if (!el) return;

    el.className = `pipeline-step-card step-${status}`;
    const statusEl    = el.querySelector('.pipeline-step-status');
    const progressBar = el.querySelector('.pipeline-step-progress-fill');
    const msgEl       = el.querySelector('.pipeline-step-msg');

    const statusLabels = {
        waiting: '等待中',
        running: '执行中...',
        done:    '完成',
        error:   '失败',
    };

    if (statusEl)    statusEl.textContent    = statusLabels[status] || status;
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (msgEl)       msgEl.textContent       = msg || '';
}

// ============================================
// 多轮迭代历史面板
// ============================================

/**
 * 清空轮次历史面板
 */
function _clearRoundsHistory() {
    const container = document.getElementById('pipeline-rounds-history');
    if (container) container.innerHTML = '';
    const section = document.getElementById('pipeline-rounds-section');
    if (section) section.style.display = 'none';
}

/**
 * 更新或追加某一轮的进度行
 * @param {Object} roundData
 */
function _upsertRoundRow(roundData) {
    const section = document.getElementById('pipeline-rounds-section');
    if (section) section.style.display = 'block';

    const container = document.getElementById('pipeline-rounds-history');
    if (!container) return;

    const rowId = `pipeline-round-row-${roundData.round}`;
    let row = document.getElementById(rowId);

    const kappa    = roundData.kappa    !== undefined ? roundData.kappa.toFixed(4)    : '-';
    const oa       = roundData.overall_accuracy !== undefined ? roundData.overall_accuracy.toFixed(2) + '%' : '-';
    const unclPct  = roundData.unclassified_ratio !== undefined ? roundData.unclassified_ratio.toFixed(1) + '%' : '-';
    const stopFlag = roundData.stop_triggered
        ? `<span class="badge badge-stop" title="${roundData.stop_reason || ''}">终止</span>`
        : '';

    const kappaClass = roundData.kappa >= 0.8 ? 'kappa-good' : (roundData.kappa >= 0.6 ? 'kappa-ok' : 'kappa-bad');

    // 生成替换值预览
    let replacementsHtml = '';
    if (roundData.auto_replacements) {
        const reps = roundData.auto_replacements;
        const repEntries = Object.entries(reps).slice(0, 4); // 最多显示4个
        replacementsHtml = repEntries.map(([name, val]) =>
            `<span class="rep-item" title="${name}">${val}</span>`
        ).join('');
        if (Object.keys(reps).length > 4) {
            replacementsHtml += `<span class="rep-item" style="opacity:0.6;">+${Object.keys(reps).length - 4}</span>`;
        }
    }

    // 类别冻结状态：显示各类别PA + 冻结徽章
    let classStatusHtml = '';
    const frozenSet = new Set(roundData.frozen_classes || []);
    if (roundData.class_accuracies && roundData.class_accuracies.length > 0) {
        classStatusHtml = roundData.class_accuracies.map(ca => {
            const isFrozen = frozenSet.has(ca.class_name);
            const frozenClass = isFrozen ? ' class-frozen' : '';
            const frozenBadge = isFrozen ? ' <span class="badge badge-stop" style="font-size:0.65rem;">冻结</span>' : '';
            const paValue = ca.producer_accuracy !== undefined ? ca.producer_accuracy.toFixed(1) + '%' : '-';
            return `<span class="round-class-pa${frozenClass}" title="PA=${paValue}">${ca.class_name} ${paValue}${frozenBadge}</span>`;
        }).join(' ');
    }

    if (!row) {
        row = document.createElement('tr');
        row.id = rowId;
        container.appendChild(row);
    }

    row.innerHTML = `
        <td class="round-num">第 ${roundData.round} 轮</td>
        <td style="font-variant-numeric:tabular-nums; color:var(--text-secondary);">${roundData.hamming_threshold !== undefined ? roundData.hamming_threshold : '-'}${roundData.distance_metric === 'euclidean' ? ' <span style="font-size:0.7rem; color:var(--text-muted);">(欧氏)</span>' : ''}</td>
        <td class="round-kappa ${kappaClass}">${kappa}</td>
        <td>${oa}</td>
        <td>${unclPct}</td>
        <td>
            <div class="round-replacements-cell">${replacementsHtml}</div>
        </td>
        <td>
            <div class="round-class-status-cell">${classStatusHtml}</div>
        </td>
        <td>${stopFlag}</td>
    `;
}

// ============================================
// 表单参数收集
// ============================================

function collectPipelineParams() {
    const get = id => {
        const el = document.getElementById(id);
        return el ? el.value.trim() : '';
    };
    const getChecked = id => {
        const el = document.getElementById(id);
        return el ? el.checked : false;
    };

    // 必填字段（目标CSV已自动推导，无需校验）
    // 影像文件夹路径始终必填：步骤3/4/5及阈值检测都依赖影像数据
    const required = [
        ['pipeline-input-img-dir', '影像文件夹路径', true],
        ['pipeline-output-dir',    '输出根目录',     true],
        ['pipeline-shp-input',     'SHP样本路径',    true],
        ['pipeline-shp-folder',    '验证SHP文件夹',  true],
    ];

    for (const [id, label, isRequired] of required) {
        if (isRequired && !get(id)) {
            alert(`请填写：${label}`);
            document.getElementById(id)?.focus();
            return null;
        }
    }

    // 地物类别（含冻结阈值配置）
    const classes = Array.from(
        document.querySelectorAll('#pipeline-classes .dynamic-item')
    ).map(item => ({
        name:   item.querySelector('.class-name')?.value   || '未命名',
        values: (item.querySelector('.class-values')?.value || '')
            .split(',')
            .map(v => parseInt(v.trim()))
            .filter(v => !isNaN(v)),
        // 冻结策略参数（兼容字段，实际由全局 consecutive_rounds 控制）
        stop_improvement_gap: 0.5,
        consecutive_rounds:   parseInt(document.getElementById('pipeline-consecutive-rounds')?.value) || 3,
    })).filter(c => c.values.length > 0);

    if (classes.length === 0) {
        // 尚未检测到类别，给出更明确的诊断信息
        const tifFolder = get('pipeline-input-img-dir');
        const shpInput  = get('pipeline-shp-input');
        let hint = '地物类别尚未自动检测。\n\n';
        if (!tifFolder) {
            hint += '❌ 「影像文件夹路径」为空\n';
        }
        if (!shpInput) {
            hint += '❌ 「SHP样本路径」为空\n';
        }
        if (tifFolder && shpInput) {
            hint += `影像路径: ${tifFolder}\nSHP路径: ${shpInput}\n\n`;
            // ★ 尝试从 placeholder 读取具体 API 错误
            const placeholder = document.getElementById('pipeline-classes-placeholder');
            if (placeholder && placeholder.textContent.trim()) {
                hint += `检测结果: ${placeholder.textContent.trim()}\n\n`;
            }
            hint += '可能原因:\n';
            hint += '1. 影像文件夹中没有 .tif/.tiff 文件\n';
            hint += '2. SHP路径中没有 .shp 文件\n';
            hint += '3. 路径不存在或权限不足\n';
            hint += '4. SHP文件名包含中文可能在 JSON 传输时有编码偏差（已在后台修复）\n';
            hint += '\n请检查路径和文件后重新填写（修改后会自动触发检测）。';
        } else {
            hint += '\n请先填写以上路径，系统将自动检测类别。';
        }
        _appendLog('warn', hint.split('\n')[0] + ' 详情请见弹窗。');
        alert(hint);
        return null;
    }

    // 波段删除范围（可选）
    const deleteModeEl = document.querySelector('#pipeline-delete-mode .radio-option.active');
    const deleteMode   = deleteModeEl ? deleteModeEl.dataset.value : 'default';
    let deleteRanges   = null;
    if (deleteMode === 'custom') {
        deleteRanges = Array.from(
            document.querySelectorAll('#pipeline-delete-ranges .dynamic-item')
        ).map(item => {
            const inputs = item.querySelectorAll('input');
            return { start: parseInt(inputs[0]?.value), end: parseInt(inputs[1]?.value) };
        }).filter(r => r.start && r.end);
        if (deleteRanges.length === 0) deleteRanges = null;
    }

    // 汉明距离阈值模式与参数
    const thresholdModeEl = document.querySelector('#pipeline-threshold-mode .radio-option.active');
    const thresholdMode = thresholdModeEl ? thresholdModeEl.dataset.value : 'auto';
    const initialThresholdEl = document.getElementById('pipeline-initial-threshold');
    const initialThreshold = initialThresholdEl ? parseFloat(initialThresholdEl.value) || 15 : 15;
    const thresholdStepEl = document.getElementById('pipeline-threshold-step');
    const thresholdStepManual = thresholdStepEl ? parseFloat(thresholdStepEl.value) || 5 : 5;
    const thresholdStepAutoEl = document.getElementById('pipeline-threshold-step-auto');
    const thresholdStepAuto = thresholdStepAutoEl ? parseFloat(thresholdStepAutoEl.value) || 5 : 5;
    const autoSamplesEl = document.getElementById('pipeline-auto-samples');
    const autoSamples = autoSamplesEl ? parseInt(autoSamplesEl.value) || 2000 : 2000;
    const thresholdStep = thresholdMode === 'manual' ? thresholdStepManual : thresholdStepAuto;

    // 消融实验2：距离度量（hamming/euclidean）
    const distanceMetric = document.querySelector('#pipeline-distance-metric .radio-option.active')?.dataset.value || 'hamming';

    // 迭代控制参数
    const kappaThreshEl    = document.getElementById('pipeline-kappa-threshold');
    const unclThreshEl     = document.getElementById('pipeline-unclassified-threshold');
    const maxIterEl        = document.getElementById('pipeline-max-iterations');
    const consecRoundsEl   = document.getElementById('pipeline-consecutive-rounds');
    const filenameEl       = document.getElementById('pipeline-filename');
    const numClassesEl     = document.getElementById('pipeline-num-classes');
    const maxSeqEl         = document.getElementById('pipeline-max-seq-length');

    // 步骤2的影像文件夹直接使用基础路径中的影像文件夹
    const inputImgDir = get('pipeline-input-img-dir');
    const outputDir   = get('pipeline-output-dir');
    const filename    = filenameEl?.value.trim() || 'merged';
    const existingStep1Csv = get('pipeline-existing-step1-csv');

    // 目标CSV文件：优先使用已有步骤1结果，否则自动推导
    const targetFile  = existingStep1Csv
        ? existingStep1Csv
        : (outputDir ? `${outputDir.replace(/\\/g, '/')}/step1/${filename}.csv` : '');

    // 同步显示到UI
    updateTargetFileDisplay(outputDir, filename);

    // 执行模式与进程数
    const execMode = document.querySelector('#pipeline-exec-mode .radio-option.active')?.dataset.value || 'multiprocess';
    const numProcSlider = document.getElementById('pipeline-num-processes');
    const numProcVal = numProcSlider ? parseInt(numProcSlider.value) : 0;
    // 滑块值 <=1 代表"自动"，传0给后端（自动 = max(8, CPU核心数)）
    const numProcesses = (execMode === 'multiprocess' && numProcVal > 1) ? numProcVal : 0;

    return {
        input_img_dir:           inputImgDir,
        output_dir:              outputDir,
        filename:                filename,
        num_classes:             parseInt(numClassesEl?.value || '5'),
        skip_discretization:     getChecked('pipeline-skip-discretization'),  // 消融实验1：跳过离散化
        delete_ranges:           deleteRanges,
        adjust_sizes:            getChecked('pipeline-adjust-sizes'),
        shp_input:               get('pipeline-shp-input'),
        tif_folder:              inputImgDir,   // 自动复用基础影像文件夹
        target_file:             targetFile,    // 自动推导
        existing_step1_csv:      existingStep1Csv || null,  // 已有步骤1结果，有值则跳过步骤1
        threshold_mode:          thresholdMode,             // 阈值模式：manual/auto
        initial_threshold:       initialThreshold,          // 初始距离阈值（手动模式下生效）
        threshold_step:          thresholdStep,             // 每轮阈值递增步长
        auto_threshold_samples:  autoSamples,             // 自动阈值采样像素数
        distance_metric:         distanceMetric,           // 消融实验2：hamming/euclidean
        execution_mode:          execMode,
        num_processes:           numProcesses,   // 进程数，0=自动
        add_geo:                 getChecked('pipeline-add-geo'),
        max_sequence_length:     maxSeqEl?.value ? parseInt(maxSeqEl.value) : null,
        classes:                 classes,
        shp_folder:              get('pipeline-shp-folder'),
        kappa_threshold:         kappaThreshEl     ? parseFloat(kappaThreshEl.value)  || 0.8  : 0.8,
        unclassified_threshold:  unclThreshEl      ? parseFloat(unclThreshEl.value)   || 10.0 : 10.0,
        max_iterations:          maxIterEl         ? parseInt(maxIterEl.value)         || 10   : 10,
        consecutive_rounds:      consecRoundsEl    ? parseInt(consecRoundsEl.value)    || 3    : 3,
        enable_freeze:           getChecked('pipeline-enable-freeze'),  // 类别冻结策略开关（默认开启）
    };
}

// ============================================
// 冻结策略开关切换
// ============================================
function toggleFreezeOption() {
    const enabled = document.getElementById('pipeline-enable-freeze')?.checked;
    const roundsInput = document.getElementById('pipeline-consecutive-rounds');
    if (roundsInput) {
        roundsInput.disabled = !enabled;   // 未启用冻结时，冻结轮次设置不可用
        roundsInput.style.opacity = enabled ? '' : '0.5';
    }
}

// ============================================
// 运行全流程
// ============================================

async function runPipeline() {
    if (pipelineRunning && !pipelinePaused) {
        alert('全流程正在执行中，请等待完成');
        return;
    }

    // ── 预检测：如果类别为空，先尝试自动检测 ──
    const existingClasses = document.querySelectorAll('#pipeline-classes .dynamic-item');
    if (existingClasses.length === 0) {
        _appendLog('info', '地物类别为空，正在尝试自动检测...');
        await loadReplacementPreview();
        // 等待一小段时间让DOM更新
        await new Promise(r => setTimeout(r, 300));
    }

    const params = collectPipelineParams();
    if (!params) return;

    pipelineRunning = true;
    pipelinePaused = false;
    pipelineResuming = false;
    currentTaskId = null;
    initPipelineSteps();
    resetTaskMonitorTiming();

    // 清空日志 & 计时
    pipelineLogs = [];
    _stepStartTimes = {};
    _lastPolledStep = 0;
    _appendLog('info', '全流程启动，准备执行...');

    // 记录实际使用的进程数配置
    {
        const _execMode  = document.querySelector('#pipeline-exec-mode .radio-option.active')?.dataset.value || 'multiprocess';
        const _numProcSl = document.getElementById('pipeline-num-processes');
        const _numProcV  = _numProcSl ? parseInt(_numProcSl.value) : 0;
        if (_execMode === 'gpu') {
            _appendLog('info', '执行模式: GPU加速 · 使用显卡批量计算汉明距离');
        } else if (_execMode === 'multiprocess') {
            const _label = (_numProcV <= 1) ? '自动（max(8, CPU核心数)）' : `${_numProcV} 进程`;
            _appendLog('info', `执行模式: 多进程并行 · 进程数: ${_label}`);
        } else {
            _appendLog('info', '执行模式: 主循环串行');
        }
    }

    // 显示进度区域
    const progressSection = document.getElementById('pipeline-progress-section');
    if (progressSection) progressSection.style.display = 'block';

    // 重置Dashboard（保持面板可见，只清空数据）
    resetDashboard();

    // 重置总消息
    const msgBox = document.getElementById('pipeline-message');
    if (msgBox) {
        msgBox.className = 'status-message';
        msgBox.innerHTML = '';
    }

    _updatePipelineButtons('running');

    // 标记步骤1为运行中
    updatePipelineStep(1, 'running', '准备中...', 0);

    try {
        // null = 不设超时，步骤2(多进程样本提取)可能需要数分钟
        const result = await API.post('/pipeline/run', params, null);
        const taskId = result.task_id;
        currentTaskId = taskId;
        _startHeartbeat(taskId);  // ★ 启动心跳
        _appendLog('info', `任务已创建，ID: ${taskId}，开始轮询状态...`);
        _pollPipelineStatus(taskId);
    } catch (error) {
        _handlePipelineError(`请求失败: ${error.message}`);
    }
}

// ============================================
// 暂停 / 恢复
// ============================================

async function pausePipeline() {
    if (!currentTaskId) return;
    try {
        await API.post(`/tasks/${currentTaskId}/pause`, {});
        pipelinePaused = true;
        _appendLog('warn', '已发送暂停请求，当前轮次完成后将暂停...');
        _updatePipelineButtons('pausing');
    } catch (err) {
        alert(`暂停失败: ${err.message}`);
    }
}

async function resumePipeline() {
    if (!currentTaskId) return;
    try {
        await API.post(`/tasks/${currentTaskId}/resume`, {});
        pipelinePaused = false;
        pipelineResuming = true;  // 标记正在恢复中
        _startHeartbeat(currentTaskId);  // ★ 恢复心跳
        _appendLog('info', '已发送恢复请求，流程继续执行...');
        _updatePipelineButtons('running');
        _pollPipelineStatus(currentTaskId);
    } catch (err) {
        alert(`恢复失败: ${err.message}`);
    }
}

async function restartPipeline() {
    if (!currentTaskId) return;
    if (!confirm('确定要重新开始吗？当前已完成的轮次数据将被保留，但流程将从头执行。')) return;
    try {
        // ★ 带重试：pipeline 可能还在步骤中尚未到达暂停检查点
        for (let attempt = 0; attempt < 8; attempt++) {
            if (attempt > 0) {
                _appendLog('info', `等待 pipeline 到达暂停点...（第${attempt}次重试）`);
                await new Promise(r => setTimeout(r, 800));
            }
            try {
                await API.post(`/tasks/${currentTaskId}/restart`, {});
                break; // 成功则跳出
            } catch (retryErr) {
                if (attempt === 7) throw retryErr; // 最后一次仍失败则抛出
            }
        }
        pipelinePaused = false;
        pipelineResuming = true;  // 标记正在重开中
        pipelineLogs = [];
        _stepStartTimes = {};
        _lastPolledStep = 0;
        _startHeartbeat(currentTaskId);  // ★ 恢复心跳
        initPipelineSteps();
        _appendLog('info', '已发送重新开始请求，流程将从步骤1重新执行...');
        _updatePipelineButtons('running');
        _pollPipelineStatus(currentTaskId);
    } catch (err) {
        const hint = `重新开始失败: ${err.message}\n\n`
            + '如果 pipeline 已完成或出错，请尝试点击"启动全流程处理"重新运行。';
        alert(hint);
    }
}

function _updatePipelineButtons(state) {
    const runBtn   = document.getElementById('pipeline-run-btn');
    const pauseBtn = document.getElementById('pipeline-pause-btn');
    const restartBtn = document.getElementById('pipeline-restart-btn');

    if (runBtn) {
        if (state === 'running') {
            runBtn.disabled = true;
            runBtn.textContent = '执行中...';
            runBtn.style.display = 'none';
        } else if (state === 'paused') {
            // 暂停时显示"恢复执行"按钮
            runBtn.disabled = false;
            runBtn.textContent = '恢复执行';
            runBtn.className = 'btn-primary btn-lg';
            runBtn.style.display = 'inline-flex';
            runBtn.onclick = resumePipeline;
        } else {
            runBtn.disabled = false;
            runBtn.textContent = '启动全流程处理';
            runBtn.className = 'btn-primary btn-lg';
            runBtn.style.display = 'inline-flex';
            runBtn.onclick = runPipeline;
        }
    }
    if (pauseBtn) {
        if (state === 'running') {
            pauseBtn.disabled = false;
            pauseBtn.style.display = 'inline-flex';
        } else if (state === 'pausing') {
            pauseBtn.disabled = true;
            pauseBtn.textContent = '暂停中...';
            pauseBtn.style.display = 'inline-flex';
        } else {
            pauseBtn.disabled = true;
            pauseBtn.textContent = '暂停';
            pauseBtn.style.display = 'none';
        }
    }
    if (restartBtn) {
        if (state === 'paused') {
            // 暂停时显示"重新开始"按钮
            restartBtn.disabled = false;
            restartBtn.style.display = 'inline-flex';
            restartBtn.className = 'btn-danger btn-lg';
        } else {
            restartBtn.disabled = true;
            restartBtn.style.display = 'none';
        }
    }
}

// ============================================
// 轮询 & UI 更新
// ============================================

function _pollPipelineStatus(taskId) {
    let _lastLogMessage = '';   // 防止重复记录相同消息
    let _404RetryCount = 0;     // 404 重试计数
    const MAX_404_RETRIES = 5;  // 最多重试 5 次（约 5 秒）
    let _timeoutRetryCount = 0; // 超时重试计数
    const MAX_TIMEOUT_RETRIES = 3;  // 超时最多重试 3 次（约 6 秒）

    const check = async () => {
        try {
            // 轮询使用 null 超时 —— 服务器负载高时状态查询可能较慢，不应因超时误杀流程
            const data = await API.get(`/tasks/${taskId}`, null);
            _404RetryCount = 0;   // 请求成功，重置重试计数
            _timeoutRetryCount = 0;  // 请求成功，重置超时重试计数
            const r = data.result || {};
            const currentStep = r.current_step || 0;
            const nowTs = Date.now();

            // ── 步骤变化检测 & 计时 ──
            if (currentStep > 0 && currentStep !== _lastPolledStep) {
                // 新步骤开始：记录开始时间
                _stepStartTimes[currentStep] = nowTs;

                // 如果之前有步骤，计算其耗时并追加日志
                if (_lastPolledStep > 0 && _stepStartTimes[_lastPolledStep]) {
                    const elapsedMs = nowTs - _stepStartTimes[_lastPolledStep];
                    const elapsedStr = _formatDuration(elapsedMs / 1000);
                    _appendLog('success', `[步骤${_lastPolledStep}] 完成 · 耗时 ${elapsedStr}`);
                }
                _lastPolledStep = currentStep;
            }

            // 收集日志（去重）
            const msg = data.message || '';
            if (msg && msg !== _lastLogMessage) {
                const step = currentStep ? `[步骤${currentStep}]` : '';
                const round = r.current_round ? `[第${r.current_round}轮]` : '';
                _appendLog('info', `${step}${round} ${msg}`);
                _lastLogMessage = msg;
            }

            if (data.status === 'pending' || data.status === 'running') {
                pipelineResuming = false;  // 成功退出恢复状态
                _updatePipelineUI(data);
                setTimeout(check, 1000);
            } else if (data.status === 'paused') {
                if (pipelineResuming) {
                    // 刚恢复/重开，后端状态尚未同步，继续轮询
                    setTimeout(check, 500);
                    return;
                }
                if (!pipelinePaused) {
                    // 首次检测到暂停，标记状态
                    pipelinePaused = true;
                    _stopHeartbeat();  // ★ 暂停时停止心跳
                }
                // 停止轮询，等待用户操作
                _updatePipelineButtons('paused');
                _appendLog('warn', `已暂停：${data.message || ''}`);
                _updatePipelineUI(data);
            } else if (data.status === 'completed') {
                // 最后一步计时
                if (_lastPolledStep > 0 && _stepStartTimes[_lastPolledStep]) {
                    const elapsedMs = nowTs - _stepStartTimes[_lastPolledStep];
                    _appendLog('success', `[步骤${_lastPolledStep}] 完成 · 耗时 ${_formatDuration(elapsedMs / 1000)}`);
                }
                _appendLog('success', '全流程处理完成！');
                _handlePipelineComplete(data);
            } else if (data.status === 'failed') {
                // 失败时记录当前步骤已运行时间
                if (_lastPolledStep > 0 && _stepStartTimes[_lastPolledStep]) {
                    const elapsedMs = nowTs - _stepStartTimes[_lastPolledStep];
                    _appendLog('warn', `[步骤${_lastPolledStep}] 中断 · 已运行 ${_formatDuration(elapsedMs / 1000)}`);
                }
                _appendLog('error', `执行失败: ${data.error || '未知错误'}`);
                _handlePipelineError(data.error || '未知错误', data.result);
            } else {
                // 未知状态也继续轮询
                setTimeout(check, 1000);
            }
        } catch (err) {
            // 超时重试逻辑：服务器负载高时 GET /api/tasks 可能响应慢
            // 与 404 不同，超时是临时状态，短暂重试后通常恢复
            if (err.message && (err.message.includes('超时') || err.message.includes('timeout') || err.name === 'AbortError')) {
                _timeoutRetryCount++;
                if (_timeoutRetryCount <= MAX_TIMEOUT_RETRIES) {
                    const delay = 1000 * _timeoutRetryCount;  // 1s/2s/3s 递增延迟
                    _appendLog('warn', `轮询超时(${_timeoutRetryCount}/${MAX_TIMEOUT_RETRIES})，${delay/1000}s后重试...`);
                    setTimeout(check, delay);
                    return;
                }
                _appendLog('error', `轮询连续超时${MAX_TIMEOUT_RETRIES}次，服务器可能过载，请稍后重试`);
                _handlePipelineError('服务器响应超时，请检查服务器状态后重新启动全流程', null);
                return;
            }

            // 404 重试逻辑：服务器可能因 auto-reload 短暂丢失任务
            if (err.message && err.message.includes('404')) {
                _404RetryCount++;
                if (_404RetryCount <= MAX_404_RETRIES) {
                    _appendLog('warn', `轮询暂时丢失任务(${_404RetryCount}/${MAX_404_RETRIES})，正在重试...`);
                    setTimeout(check, 1000);
                    return;
                }
                _appendLog('error', '任务丢失：服务器可能已重启，请重新启动全流程');
                _handlePipelineError('任务丢失（服务器可能已重启），请重新启动全流程', null);
                return;
            }
            _appendLog('error', `轮询异常: ${err.message}`);
            _handlePipelineError(err.message);
        }
    };
    check();
}

/**
 * 运行中的UI更新（含多轮状态）
 */
function _updatePipelineUI(data) {
    const r          = data.result || {};
    const currentStep = r.current_step || 1;
    const stepMsg     = r.step_message || data.message || '';
    const stepPct     = r.step_progress || 0;
    const currentRound = r.current_round || 0;

    // 步骤1-2：固定步骤
    if (currentStep <= 2) {
        for (let i = 1; i < currentStep; i++) {
            updatePipelineStep(i, 'done', '完成', 100);
        }
        updatePipelineStep(currentStep, 'running', stepMsg, stepPct);
        // 步骤3-6暂时等待
        for (let i = currentStep + 1; i <= 6; i++) {
            updatePipelineStep(i, 'waiting', '', 0);
        }
    } else {
        // 步骤1-2完成
        updatePipelineStep(1, 'done', '完成', 100);
        updatePipelineStep(2, 'done', '完成', 100);

        // 步骤3-6在每轮迭代中循环运行
        // 当前步骤标记运行，小于当前步骤的3-6标完成，大于的等待
        for (let s = 3; s <= 6; s++) {
            if (s < currentStep) {
                updatePipelineStep(s, 'done', `${r.current_round ? `第${r.current_round}轮` : ''}完成`, 100);
            } else if (s === currentStep) {
                updatePipelineStep(s, 'running', stepMsg, stepPct);
            } else {
                updatePipelineStep(s, 'waiting', '', 0);
            }
        }
    }

    // 更新轮次标题
    if (currentRound > 0) {
        const roundLabel = document.getElementById('pipeline-current-round-label');
        if (roundLabel) roundLabel.textContent = `当前迭代：第 ${currentRound} 轮`;
    }

    // 同步已完成的轮次记录到历史表格
    if (r.all_rounds && r.all_rounds.length > 0) {
        r.all_rounds.forEach(rd => _upsertRoundRow(rd));
        // 同时更新Dashboard精度曲线（若图表未初始化则先初始化）
        if (!dashAccuracyChart) {
            renderDashboardAccuracyChart(r.all_rounds);
        } else {
            updateDashboardAccuracyChart(r.all_rounds);
        }
    }

    // 步骤2完成后，保存光谱指纹数据并渲染
    if (r.spectral_profiles && Object.keys(r.spectral_profiles).length > 0) {
        dashSpectralData = r.spectral_profiles;
        renderDashboardSpectral();
    }

    // 更新Dashboard流程状态
    updateDashboardProcess(data, r);

    // 更新Dashboard指标卡片（步骤1/2后静态指标可用，步骤6后动态指标更新）
    renderDashboardMetrics(null, r);

    // 更新总进度
    const totalFill = document.getElementById('pipeline-total-progress-fill');
    const totalPct  = document.getElementById('pipeline-total-percent');
    const totalMsg  = document.getElementById('pipeline-total-status');
    if (totalFill) totalFill.style.width = `${data.progress || 0}%`;
    if (totalPct)  totalPct.textContent  = `${Math.round(data.progress || 0)}%`;
    if (totalMsg)  totalMsg.textContent  = data.message || '';

    showTaskMonitor(data.message || '', data.progress || 0, r.timing || null);
}

/**
 * 全流程完成
 */
function _handlePipelineComplete(data) {
    pipelineRunning = false;
    pipelinePaused = false;
    pipelineResuming = false;
    currentTaskId = null;
    _stopHeartbeat();  // ★ 停止心跳
    _updatePipelineButtons('idle');

    const r = data.result || {};

    // 所有步骤标绿
    PIPELINE_STEPS.forEach(s => updatePipelineStep(s.id, 'done', '完成', 100));

    // 总进度100%
    const totalFill = document.getElementById('pipeline-total-progress-fill');
    const totalPct  = document.getElementById('pipeline-total-percent');
    const totalMsg  = document.getElementById('pipeline-total-status');
    if (totalFill) totalFill.style.width = '100%';
    if (totalPct)  totalPct.textContent  = '100%';
    if (totalMsg)  totalMsg.textContent  = '全流程处理完成';

    // 轮次历史
    if (r.all_rounds && r.all_rounds.length > 0) {
        r.all_rounds.forEach(rd => _upsertRoundRow(rd));
    }

    // 显示终止原因横幅
    if (r.stop_reason) {
        const banner = document.getElementById('pipeline-stop-banner');
        if (banner) {
            banner.textContent = `✔ ${r.stop_reason}（共 ${r.total_rounds || 1} 轮）`;
            banner.style.display = 'block';
        }
    }

    // 显示最终精度结果
    _renderPipelineResults(r);

    // 成功消息
    const msgBox = document.getElementById('pipeline-message');
    if (msgBox) {
        msgBox.className = 'status-message active status-success';
        msgBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> 全流程处理完成！共 ${r.total_rounds || 1} 轮`;
    }

    // 显示计时总结（若有）
    const timing = r.timing || {};
    if (timing.overall_seconds) {
        let timingHtml = `<strong>总运行时间:</strong> ${_formatDuration(timing.overall_seconds)}<br>`;
        if (timing.round_times && timing.round_times.length > 0) {
            timingHtml += `<strong>每轮耗时:</strong> `;
            timingHtml += timing.round_times.map((t, i) => `第${i + 1}轮 ${_formatDuration(t)}`).join(' / ');
            timingHtml += '<br>';
        }
        if (timing.stage_times) {
            const st = timing.stage_times;
            const stageLabels = { 'step1': '步骤1', 'step2': '步骤2', 'step3': '步骤3', 'step4': '步骤4', 'step5': '步骤5', 'step6': '步骤6' };
            const stageParts = Object.entries(st).map(([k, v]) => `${stageLabels[k] || k} ${_formatDuration(v)}`);
            if (stageParts.length > 0) {
                timingHtml += `<strong>各阶段耗时:</strong> ${stageParts.join(' / ')}<br>`;
            }
        }
        timingHtml += `<span style="color:var(--text-muted)">已保存至 ${timing.report_file || 'timing_report.txt'}</span>`;
        showTaskMonitor('全流程处理完成！', 100, null, true, timingHtml);
    } else {
        hideTaskMonitor();
    }
}

/**
 * 全流程失败
 */
function _handlePipelineError(errMsg, resultData) {
    pipelineRunning = false;
    pipelinePaused = false;
    pipelineResuming = false;
    currentTaskId = null;
    _stopHeartbeat();  // ★ 停止心跳
    _updatePipelineButtons('idle');

    // 已完成的轮次依然渲染
    if (resultData?.all_rounds?.length > 0) {
        resultData.all_rounds.forEach(rd => _upsertRoundRow(rd));
    }

    PIPELINE_STEPS.forEach(s => {
        const el = document.getElementById(`pipeline-step-${s.id}`);
        if (el && el.classList.contains('step-running')) {
            updatePipelineStep(s.id, 'error', errMsg, 0);
        }
    });

    const msgBox = document.getElementById('pipeline-message');
    if (msgBox) {
        msgBox.className = 'status-message active status-error';
        msgBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> 失败: ${errMsg}`;
    }

    // 失败时如有计时数据也展示
    const timing = resultData?.timing;
    if (timing && timing.overall_seconds) {
        let timingHtml = `<strong>已运行:</strong> ${_formatDuration(timing.overall_seconds)}（失败前）`;
        showTaskMonitor('处理失败', 0, null, true, timingHtml);
    } else {
        hideTaskMonitor();
    }
}

// ============================================
// 结果渲染
// ============================================

function _renderPipelineResults(result) {
    const resultsDiv = document.getElementById('pipeline-results');
    if (!resultsDiv) return;

    const metrics = result.metrics;
    if (!metrics) {
        resetDashboard();
        return;
    }

    // 渲染完整Dashboard
    renderDashboard(result);
}

function _outputLabel(key) {
    const labels = {
        step1_csv:        '步骤1 CSV',
        step2_dir:        '步骤2 样本目录',
        final_round:      '最终迭代轮次',
        final_round_dir:  '最终输出目录',
    };
    return labels[key] || key;
}

// ============================================
// 动态表单辅助（地物类别 & 删除波段 & 轮次）
// ============================================

function addPipelineClass(name = '', values = '') {
    const container = document.getElementById('pipeline-classes');
    if (!container) return;

    // 首次添加时移除占位提示
    const placeholder = document.getElementById('pipeline-classes-placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'dynamic-item class-item auto-detected';
    item.innerHTML = `
        <div class="class-item-row">
            <input type="text" class="class-name" placeholder="类别名称" style="flex:2;" readonly title="由步骤3自动检测">
            <input type="text" class="class-values" placeholder="数值列表" style="flex:2;" readonly title="由步骤3自动分配">
        </div>
        <span class="badge badge-info" style="font-size:0.65rem; padding:0.15rem 0.4rem; white-space:nowrap;">自动</span>
    `;
    const nameInput = item.querySelector('.class-name');
    const valuesInput = item.querySelector('.class-values');
    if (nameInput) nameInput.value = name;
    if (valuesInput) valuesInput.value = values;
    container.appendChild(item);
}

// ============================================
// 步骤1 CSV 来源模式切换
// ============================================

/**
 * 切换步骤1 CSV 来源模式：none（重新运行）/ scan（扫描）/ manual（手动）
 */
function setStep1CsvMode(el, mode) {
    // 切换 radio 高亮
    document.querySelectorAll('#step1-csv-source-mode .radio-option').forEach(r => r.classList.remove('active'));
    if (el) el.classList.add('active');

    const scanPanel   = document.getElementById('step1-csv-scan-panel');
    const manualPanel = document.getElementById('step1-csv-manual-panel');
    const hidden      = document.getElementById('pipeline-existing-step1-csv');
    const hint        = document.getElementById('pipeline-existing-step1-hint');

    // 隐藏全部面板
    if (scanPanel)   scanPanel.style.display   = 'none';
    if (manualPanel) manualPanel.style.display = 'none';

    if (mode === 'none') {
        // 重新运行：清空隐藏值
        if (hidden) hidden.value = '';
        if (hint)   hint.style.display = 'none';
        _syncImgDirLabel('');
    } else if (mode === 'scan') {
        if (scanPanel) scanPanel.style.display = '';
        // 同步输出目录输入框内容到扫描目录
        const outDir = document.getElementById('pipeline-output-dir')?.value?.trim() || '';
        const scanInput = document.getElementById('step1-csv-scan-dir');
        if (scanInput && outDir && !scanInput.value) scanInput.value = outDir;
    } else if (mode === 'manual') {
        if (manualPanel) manualPanel.style.display = '';
    }
}

/**
 * 扫描输出目录下的 step1 CSV 文件
 */
async function scanStep1CsvFiles() {
    const scanInput = document.getElementById('step1-csv-scan-dir');
    const listDiv   = document.getElementById('step1-csv-scan-list');
    const itemsDiv  = document.getElementById('step1-csv-scan-items');
    const emptyTip  = document.getElementById('step1-csv-scan-empty');

    if (!scanInput || !scanInput.value.trim()) {
        alert('请先填写输出目录路径');
        return;
    }
    const dir = scanInput.value.trim();

    // 按钮状态
    const btn = scanInput.parentElement?.querySelector('button');
    if (btn) { btn.disabled = true; btn.textContent = '扫描中…'; }

    try {
        const resp = await fetch(`/api/files/scan-step1-csv?output_dir=${encodeURIComponent(dir)}`);
        const data = await resp.json();

        if (listDiv) listDiv.style.display = '';
        if (itemsDiv) itemsDiv.innerHTML = '';

        if (!data.exists || !data.files || data.files.length === 0) {
            if (emptyTip) emptyTip.style.display = '';
        } else {
            if (emptyTip) emptyTip.style.display = 'none';
            data.files.forEach(f => {
                const item = document.createElement('div');
                item.style.cssText = 'display:flex; align-items:center; gap:8px; padding:5px 10px; cursor:pointer; border-radius:4px; transition:background .15s;';
                item.addEventListener('mouseenter', () => item.style.background = 'var(--bg-hover, rgba(255,255,255,.06))');
                item.addEventListener('mouseleave', () => item.style.background = '');

                const rowColStr = (f.rows != null && f.cols != null) ? `${f.rows}行 × ${f.cols}列` : '（无法解析）';
                item.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14" style="flex-shrink:0;opacity:.6">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span style="flex:1; font-size:.85rem; font-weight:500;">${f.name}</span>
                    <span style="font-size:.75rem; opacity:.55;">${rowColStr}</span>
                    <span style="font-size:.75rem; opacity:.45;">${f.size_mb} MB</span>
                `;
                item.addEventListener('click', () => selectStep1CsvFile(f.path, f.name, rowColStr));
                itemsDiv.appendChild(item);
            });
        }
    } catch(e) {
        alert('扫描失败：' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '扫描'; }
    }
}

/**
 * 从扫描列表中选中一个 CSV 文件
 */
function selectStep1CsvFile(path, name, rowColStr) {
    const hidden = document.getElementById('pipeline-existing-step1-csv');
    const hint   = document.getElementById('pipeline-existing-step1-hint');
    const items  = document.getElementById('step1-csv-scan-items');

    if (hidden) hidden.value = path;

    // 更新列表高亮
    if (items) {
        items.querySelectorAll('div').forEach(d => d.style.background = '');
        // 找到对应项高亮
        items.querySelectorAll('div').forEach(d => {
            if (d.querySelector('span')?.textContent?.trim() === name) {
                d.style.background = 'var(--color-primary-10, rgba(99,102,241,.15))';
            }
        });
    }

    if (hint) {
        hint.style.display = '';
        hint.textContent = `✓ 已选：${name}（${rowColStr}），将跳过步骤1直接从步骤2开始运行`;
    }
    _syncImgDirLabel(path);
}

/**
 * 根据是否选中已有CSV同步影像目录标签显示
 */
function _syncImgDirLabel(csvVal) {
    const imgDirGroup = document.getElementById('pipeline-input-img-dir')?.closest('.form-group');
    const imgDirLabel = imgDirGroup?.querySelector('.form-label');
    if (!imgDirLabel) return;
    if (csvVal) {
        imgDirLabel.querySelectorAll('.badge-required, .badge-optional-note').forEach(el => el.remove());
        if (!imgDirLabel.querySelector('.badge-optional-note')) {
            const optNote = document.createElement('span');
            optNote.className = 'badge badge-info badge-optional-note';
            optNote.style.cssText = 'font-size:0.7rem; padding:0.1rem 0.4rem; margin-left:0.4rem;';
            optNote.textContent = '可选（已跳过步骤1）';
            imgDirLabel.appendChild(optNote);
        }
    } else {
        imgDirLabel.querySelectorAll('.badge-optional-note').forEach(el => el.remove());
        if (!imgDirLabel.querySelector('.badge-required')) {
            const req = document.createElement('span');
            req.className = 'badge-required';
            req.textContent = '必填';
            imgDirLabel.appendChild(req);
        }
    }
}



/**
 * 手动路径输入框的 oninput 回调：同步到隐藏字段
 */
function onExistingStep1CsvChange() {
    const manualInput = document.getElementById('step1-csv-manual-path');
    const hidden      = document.getElementById('pipeline-existing-step1-csv');
    const hint        = document.getElementById('pipeline-existing-step1-hint');

    if (!manualInput) return;
    const val = manualInput.value.trim();
    if (hidden) hidden.value = val;

    if (val) {
        if (hint) { hint.style.display = ''; hint.textContent = `✓ 已指定步骤1结果文件，将跳过步骤1直接从步骤2开始运行`; }
        _syncImgDirLabel(val);
    } else {
        if (hint) hint.style.display = 'none';
        _syncImgDirLabel('');
    }
}

function addPipelineDeleteRange() {
    const container = document.getElementById('pipeline-delete-ranges');
    if (!container) return;
    const item = document.createElement('div');
    item.className = 'dynamic-item';
    item.innerHTML = `
        <input type="number" placeholder="起始波段" min="1">
        <span>-</span>
        <input type="number" placeholder="结束波段" min="1">
        <button type="button" class="btn-icon" onclick="this.parentElement.remove()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    `;
    container.appendChild(item);
}


function setPipelineDeleteMode(el, mode) {
    document.querySelectorAll('#pipeline-delete-mode .radio-option').forEach(r => r.classList.remove('active'));
    el.classList.add('active');
    const customSection = document.getElementById('pipeline-custom-delete-section');
    if (customSection) {
        customSection.style.display = mode === 'custom' ? 'block' : 'none';
    }
}

function setExecMode(mode) {
    document.querySelectorAll('#pipeline-exec-mode .radio-option').forEach(r => {
        r.classList.toggle('active', r.dataset.value === mode);
    });
    // 根据模式显示/隐藏进程数设置
    const numProcGroup = document.getElementById('pipeline-num-processes-group');
    if (numProcGroup) {
        numProcGroup.style.display = mode === 'multiprocess' ? '' : 'none';
    }
    // 更新提示文字
    const hint = document.getElementById('exec-mode-hint');
    if (hint) {
        if (mode === 'gpu') {
            hint.textContent = 'GPU加速模式利用显卡并行计算汉明距离矩阵，大规模数据可提速5-8倍';
        } else if (mode === 'multiprocess') {
            hint.textContent = '多进程模式利用多核CPU并行匹配，速度较快且兼容性好';
        } else {
            hint.textContent = '串行模式仅用单核CPU，速度较慢但更稳定，适合调试';
        }
    }
}

// ═══════════════════════════════════════════════════════════
// GPU状态检测
// ═══════════════════════════════════════════════════════════

async function checkGPUStatus() {
    const badge = document.getElementById('gpu-status-badge');
    const gpuLabel = document.getElementById('gpu-option-label');
    if (!badge) return;

    try {
        const data = await API.get('/gpu/status');
        if (data.gpu_available) {
            badge.textContent = `GPU: ${data.gpu_name}`;
            badge.style.cssText = 'font-size:0.65rem; padding:0.1rem 0.4rem; margin-left:0.4rem; border-radius:4px; background:#10b981; color:#fff; display:inline-block;';
            badge.style.display = 'inline-block';

            // 启用GPU选项并设为默认
            if (gpuLabel) {
                gpuLabel.style.opacity = '1';
                gpuLabel.style.cursor = 'pointer';
            }
            // 自动切换到GPU模式
            const currentMode = document.querySelector('#pipeline-exec-mode .radio-option.active')?.dataset.value;
            if (currentMode !== 'gpu') {
                setExecMode('gpu');
            }
            console.log(`[GPU] GPU可用: ${data.gpu_name}, 显存: ${data.vram_free_mb}MB/${data.vram_total_mb}MB`);
        } else {
            badge.textContent = 'GPU不可用';
            badge.style.cssText = 'font-size:0.65rem; padding:0.1rem 0.4rem; margin-left:0.4rem; border-radius:4px; background:#6b7280; color:#fff; display:inline-block;';
            badge.style.display = 'inline-block';

            // 禁用GPU选项
            if (gpuLabel) {
                gpuLabel.style.opacity = '0.5';
                gpuLabel.style.cursor = 'not-allowed';
                gpuLabel.onclick = function(e) {
                    e.preventDefault();
                    alert('GPU不可用。请确保已安装CuPy (pip install cupy-cuda12x[ctk]) 且有NVIDIA显卡。');
                };
            }
            console.log('[GPU] GPU不可用:', data.reason || '未知原因');
        }
    } catch (e) {
        badge.textContent = 'GPU检测失败';
        badge.style.cssText = 'font-size:0.65rem; padding:0.1rem 0.4rem; margin-left:0.4rem; border-radius:4px; background:#ef4444; color:#fff; display:inline-block;';
        badge.style.display = 'inline-block';
        if (gpuLabel) {
            gpuLabel.style.opacity = '0.5';
        }
        console.warn('[GPU] 状态检测失败:', e);
    }
}

function setThresholdMode(mode) {
    document.querySelectorAll('#pipeline-threshold-mode .radio-option').forEach(r => {
        r.classList.toggle('active', r.dataset.value === mode);
    });
    const manualSettings = document.getElementById('threshold-manual-settings');
    const autoSettings = document.getElementById('threshold-auto-settings');
    if (manualSettings) manualSettings.style.display = mode === 'manual' ? 'block' : 'none';
    if (autoSettings) autoSettings.style.display = mode === 'auto' ? 'block' : 'none';
}

/**
 * 消融实验2：设置距离度量（hamming=汉明距离 / euclidean=欧氏距离）
 */
function setDistanceMetric(metric) {
    document.querySelectorAll('#pipeline-distance-metric .radio-option').forEach(r => {
        r.classList.toggle('active', r.dataset.value === metric);
    });
    // 更新阈值相关提示文字（欧氏距离阈值语义为连续距离值）
    const thresholdLabel = document.querySelector('#pipeline-threshold-mode')?.closest('.form-group')?.querySelector('.form-label');
    if (thresholdLabel) {
        thresholdLabel.textContent = metric === 'euclidean' ? '距离阈值模式' : '汉明距离阈值模式';
    }
    const hint = document.querySelector('#pipeline-distance-metric')?.closest('.form-group')?.querySelector('.form-hint');
    if (hint) {
        if (metric === 'euclidean') {
            hint.textContent = '🔬 欧氏距离模式：按连续数值计算序列间距离（消融实验对比项）。阈值语义为连续距离值，建议配合自动阈值模式';
        } else {
            hint.textContent = '🔬 汉明距离=位级匹配（默认）；欧氏距离=连续数值距离匹配（消融实验对比项）。选择欧氏距离时阈值语义为连续距离值，建议配合自动阈值模式';
        }
    }
}

/**
 * 更新进程数显示（滑块回调）
 * value=0 表示"自动"（滑块实际最小值为1，但我们用value=0代表auto）
 * 滑块范围: 1-32，拖到1时我们额外检测是否意味着"自动"
 */
function updateNumProcessesDisplay(value) {
    const val = parseInt(value);
    const displayEl = document.getElementById('pipeline-num-processes-display');
    const badgeEl   = document.getElementById('pipeline-num-processes-badge');

    if (val <= 1) {
        // 最左端 = 自动
        if (displayEl) displayEl.textContent = '自动';
        if (badgeEl)   { badgeEl.textContent = '自动'; badgeEl.style.background = ''; }
    } else {
        if (displayEl) displayEl.textContent = `${val} 进程`;
        if (badgeEl)   { badgeEl.textContent = `${val} 进程`; badgeEl.style.background = 'rgba(99,102,241,0.12)'; }
    }
}

// ============================================
// 目标CSV文件自动显示更新
// ============================================

function updateTargetFileDisplay(outputDir, filename) {
    const display = document.getElementById('pipeline-target-file-display');
    if (!display) return;
    const fname = (filename || 'merged').trim();
    if (!outputDir || !outputDir.trim()) {
        display.innerHTML = '<span class="text-muted">请先填写「输出根目录」和「输出文件名」</span>';
        return;
    }
    const path = `${outputDir.replace(/\\/g, '/')}/step1/${fname}.csv`;
    display.textContent = path;
    display.style.color = 'var(--text-primary)';
}

// ============================================
// 事件监听器初始化
// ============================================

function initPipelineEventListeners() {
    // 监听影像文件夹和SHP路径的变化，自动加载替换值预览
    const imgDirInput = document.getElementById('pipeline-input-img-dir');
    const shpInputInput = document.getElementById('pipeline-shp-input');

    if (imgDirInput) {
        imgDirInput.addEventListener('change', debounce(loadReplacementPreview, 500));
        imgDirInput.addEventListener('input', debounce(loadReplacementPreview, 800));
    }

    if (shpInputInput) {
        shpInputInput.addEventListener('change', debounce(loadReplacementPreview, 500));
        shpInputInput.addEventListener('input', debounce(loadReplacementPreview, 800));
    }

    // 监听输出目录和文件名变化，实时更新目标CSV显示
    const outputDirInput = document.getElementById('pipeline-output-dir');
    const filenameInput  = document.getElementById('pipeline-filename');

    const refreshTargetDisplay = () => {
        const outDir = outputDirInput?.value?.trim() || '';
        const fname  = filenameInput?.value?.trim() || 'merged';
        updateTargetFileDisplay(outDir, fname);
    };

    if (outputDirInput) {
        outputDirInput.addEventListener('input', debounce(refreshTargetDisplay, 300));
        outputDirInput.addEventListener('change', () => {
            refreshTargetDisplay();
            loadReplacementPreview();
        });
        // 同步输出目录到扫描面板的目录输入框
        outputDirInput.addEventListener('input', () => {
            const scanInput = document.getElementById('step1-csv-scan-dir');
            if (scanInput && !scanInput.value) {
                scanInput.value = outputDirInput.value.trim();
            }
        });
    }
    if (filenameInput) {
        filenameInput.addEventListener('input', debounce(refreshTargetDisplay, 300));
        filenameInput.addEventListener('change', () => {
            refreshTargetDisplay();
            loadReplacementPreview();
        });
    }

    // 初始化Dashboard为空状态
    resetDashboard();

    // 初始化进程数滑块显示
    const numProcSlider = document.getElementById('pipeline-num-processes');
    if (numProcSlider) {
        updateNumProcessesDisplay(numProcSlider.value);
    }

    // 初始化执行模式显示（确保进程数面板状态与默认选中模式一致）
    const activeMode = document.querySelector('#pipeline-exec-mode .radio-option.active')?.dataset.value || 'multiprocess';
    const numProcGroup = document.getElementById('pipeline-num-processes-group');
    if (numProcGroup) {
        numProcGroup.style.display = activeMode === 'multiprocess' ? '' : 'none';
    }

    // 页面加载时若已填写路径，自动触发类别检测
    const imgDirVal = imgDirInput?.value?.trim();
    const shpVal    = shpInputInput?.value?.trim();
    if (imgDirVal && shpVal) {
        loadReplacementPreview();
    }
}

/**
 * 防抖函数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================
// Dashboard 结果展示面板
// ============================================

function resetDashboard() {
    dashRoundHistory = [];
    dashProcessSteps = [];

    // 指标卡片
    const ids = ['dash-bands', 'dash-image-size', 'dash-class-count', 'dash-sample-count', 'dash-oa', 'dash-kappa'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '--';
    });

    // 分类结果图占位
    const mapPlaceholder = document.getElementById('dash-classification-map');
    if (mapPlaceholder) {
        mapPlaceholder.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
            <p class="text-muted">运行后将显示分类结果</p>
        `;
    }

    // 图例
    const legend = document.getElementById('dash-legend');
    if (legend) legend.innerHTML = '';

    // 精度表格
    const tableBody = document.getElementById('dash-class-table');
    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-muted" style="text-align:center; padding:2rem;">运行后显示数据</td></tr>';
    }

    // 精度曲线
    if (dashAccuracyChart) {
        dashAccuracyChart.destroy();
        dashAccuracyChart = null;
    }
    const accCanvas = document.getElementById('dash-accuracy-chart');
    if (accCanvas) {
        const ctx = accCanvas.getContext('2d');
        ctx.clearRect(0, 0, accCanvas.width, accCanvas.height);
    }

    // 光谱曲线
    dashSpectralData = null;
    if (dashSpectralChart) {
        dashSpectralChart.destroy();
        dashSpectralChart = null;
    }
    const specCanvas = document.getElementById('dash-spectral-chart');
    if (specCanvas) {
        const ctx = specCanvas.getContext('2d');
        ctx.clearRect(0, 0, specCanvas.width, specCanvas.height);
        ctx.fillStyle = getComputedStyle(document.body).color || '#71717a';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('运行后将显示各类别光谱指纹曲线', specCanvas.width / 2, specCanvas.height / 2);
    }
    const specLegend = document.getElementById('dash-spectral-legend');
    if (specLegend) specLegend.innerHTML = '';

    // 处理流程
    const processList = document.getElementById('dash-process-status');
    if (processList) processList.innerHTML = '';

    // 输出路径
    const outputsList = document.getElementById('pipeline-outputs');
    if (outputsList) outputsList.innerHTML = '';
}

function renderDashboard(result) {
    const metrics  = result.metrics;
    const classAcc = result.class_accuracies;
    const outputs  = result.outputs || {};
    const allRounds = result.all_rounds || [];

    renderDashboardMetrics(metrics, result);
    renderDashboardAccuracyChart(allRounds);
    renderDashboardClassTable(classAcc);
    renderDashboardSpectral();
    renderDashboardOutputs(outputs);
    // 流程状态由 _updatePipelineUI / updateDashboardProcess 实时维护
}

function renderDashboardMetrics(metrics, result) {
    const dm = result.dashboard_metrics || {};

    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val ?? '--';
    };

    // 静态指标（步骤1/2后填充，存于 dashboard_metrics）
    setText('dash-bands',        dm.bands_count ?? '--');
    setText('dash-image-size',   dm.image_size ?? '--');
    setText('dash-class-count',  dm.class_count ?? '--');
    setText('dash-sample-count', dm.sample_count ?? '--');

    // 动态指标（每轮步骤6后更新，优先从 dashboard_metrics 读取，回退到 metrics）
    const oa  = dm.overall_accuracy ?? metrics?.overall_accuracy ?? '--';
    const kap = dm.kappa ?? metrics?.kappa ?? '--';
    setText('dash-oa',    oa  !== '--' ? oa + '%' : '--');
    setText('dash-kappa', kap !== '--' ? kap : '--');
}

function renderDashboardAccuracyChart(allRounds) {
    if (!allRounds || allRounds.length === 0) return;

    const canvas = document.getElementById('dash-accuracy-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const labels = allRounds.map((_, i) => `第${i + 1}轮`);
    const kappaData = allRounds.map(r => typeof r.kappa === 'number' ? r.kappa : null);
    const oaData    = allRounds.map(r => typeof r.overall_accuracy === 'number' ? r.overall_accuracy : null);

    if (dashAccuracyChart) dashAccuracyChart.destroy();

    const isDark = document.body.classList.contains('dark') ||
        getComputedStyle(document.body).backgroundColor !== 'rgb(255, 255, 255)';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#a1a1aa' : '#666';

    dashAccuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Kappa系数',
                    data: kappaData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,0.08)',
                    tension: 0.35,
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                },
                {
                    label: '总体精度 OA',
                    data: oaData,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34,197,94,0.08)',
                    tension: 0.35,
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    hidden: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: textColor, font: { size: 12 } },
                },
                tooltip: {
                    backgroundColor: isDark ? 'rgba(24,24,27,0.95)' : 'rgba(255,255,255,0.95)',
                    titleColor: isDark ? '#fafafa' : '#111',
                    bodyColor: isDark ? '#a1a1aa' : '#444',
                    borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { size: 11 } },
                },
                y: {
                    min: 0,
                    max: 1.05,
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { size: 11 } },
                },
            },
        },
    });

    // 下拉切换 + 初始状态同步
    const select = document.getElementById('dash-metric-select');
    if (select) {
        const syncVisibility = () => {
            const val = select.value;
            if (dashAccuracyChart) {
                dashAccuracyChart.data.datasets[0].hidden = val !== 'kappa';
                dashAccuracyChart.data.datasets[1].hidden = val !== 'oa';
                dashAccuracyChart.update();
            }
        };
        select.onchange = syncVisibility;
        syncVisibility(); // 初始同步：确保图表与下拉框当前选项一致
    }
}

function updateDashboardAccuracyChart(allRounds) {
    if (!allRounds || allRounds.length === 0 || !dashAccuracyChart) return;
    const labels = allRounds.map((_, i) => `第${i + 1}轮`);
    const kappaData = allRounds.map(r => typeof r.kappa === 'number' ? r.kappa : null);
    const oaData    = allRounds.map(r => typeof r.overall_accuracy === 'number' ? r.overall_accuracy : null);
    dashAccuracyChart.data.labels = labels;
    dashAccuracyChart.data.datasets[0].data = kappaData;
    dashAccuracyChart.data.datasets[1].data = oaData;
    dashAccuracyChart.update('none');
}

function renderDashboardClassTable(classAcc) {
    const tbody = document.getElementById('dash-class-table');
    if (!tbody || !classAcc || classAcc.length === 0) return;

    tbody.innerHTML = classAcc.map(c => {
        const pa = c.producer_accuracy ?? c.pa ?? '--';
        const ua = c.user_accuracy ?? c.ua ?? '--';
        const f1 = c.f1_score ?? c.f1 ?? '--';
        const count = c.sample_count ?? c.count ?? '--';
        const color = c.color || '#14b8a6';
        return `
            <tr>
                <td>
                    <span class="dashboard-legend-dot" style="background:${color}; display:inline-block; vertical-align:middle; margin-right:0.35rem;"></span>
                    ${c.class_name}
                </td>
                <td>${count}</td>
                <td>${typeof pa === 'number' ? pa.toFixed(2) + '%' : pa}</td>
                <td>${typeof ua === 'number' ? ua.toFixed(2) + '%' : ua}</td>
                <td>${typeof f1 === 'number' ? f1.toFixed(2) : f1}</td>
            </tr>
        `;
    }).join('');
}

function renderDashboardSpectral() {
    const canvas = document.getElementById('dash-spectral-chart');
    if (!canvas) return;

    // 如果后端未提供光谱数据，显示空占位
    if (!dashSpectralData || Object.keys(dashSpectralData).length === 0) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = getComputedStyle(document.body).color || '#71717a';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('步骤2完成后将显示各类别光谱指纹曲线', canvas.width / 2, canvas.height / 2);
        return;
    }

    const ctx = canvas.getContext('2d');
    if (dashSpectralChart) dashSpectralChart.destroy();

    const isDark = document.body.classList.contains('dark') ||
        getComputedStyle(document.body).backgroundColor !== 'rgb(255, 255, 255)';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#a1a1aa' : '#666';

    // 颜色列表
    const colors = [
        '#3b82f6', '#22c55e', '#f59e0b', '#ef4444',
        '#a855f7', '#06b6d4', '#f97316', '#ec4899',
    ];

    // 构建数据集
    const classNames = Object.keys(dashSpectralData);
    const numBands = dashSpectralData[classNames[0]]?.length || 0;
    const labels = Array.from({ length: numBands }, (_, i) => `B${i + 1}`);

    const datasets = classNames.map((name, idx) => ({
        label: name,
        data: dashSpectralData[name],
        borderColor: colors[idx % colors.length],
        backgroundColor: colors[idx % colors.length] + '20',
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
        fill: false,
    }));

    dashSpectralChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        boxWidth: 8,
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                tooltip: {
                    backgroundColor: isDark ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.95)',
                    titleColor: isDark ? '#fff' : '#333',
                    bodyColor: isDark ? '#ddd' : '#555',
                    borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                    borderWidth: 1,
                    padding: 8,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: gridColor, drawBorder: false },
                    ticks: { color: textColor, font: { size: 10 } },
                    title: { display: true, text: '波段', color: textColor, font: { size: 11 } },
                },
                y: {
                    grid: { color: gridColor, drawBorder: false },
                    ticks: { color: textColor, font: { size: 10 } },
                    title: { display: true, text: '光谱值', color: textColor, font: { size: 11 } },
                },
            },
        },
    });

    // 更新图例区域
    const legendEl = document.getElementById('dash-spectral-legend');
    if (legendEl) {
        legendEl.innerHTML = classNames.map((name, idx) => `
            <span class="legend-item" style="color:${colors[idx % colors.length]}">
                <span class="legend-dot" style="background:${colors[idx % colors.length]}"></span>
                ${name}
            </span>
        `).join('');
    }
}

function renderDashboardOutputs(outputs) {
    const outputsList = document.getElementById('pipeline-outputs');
    if (!outputsList) return;
    if (Object.keys(outputs).length === 0) {
        outputsList.innerHTML = '';
        return;
    }
    outputsList.innerHTML = Object.entries(outputs).map(([k, v]) => `
        <div class="output-item">
            <span class="output-label">${_outputLabel(k)}</span>
            <span class="output-path">${v}</span>
        </div>
    `).join('');
}

function updateDashboardProcess(data, result) {
    const list = document.getElementById('dash-process-status');
    if (!list) return;

    const steps = [
        { key: 'step1', name: '影像数据加载', icon: 'image' },
        { key: 'step2', name: '光谱指纹提取', icon: 'zap' },
        { key: 'step3', name: '样本库构建', icon: 'database' },
        { key: 'step4', name: '光谱匹配分类', icon: 'activity' },
        { key: 'step5', name: '结果后处理', icon: 'cpu' },
        { key: 'step6', name: '精度评价', icon: 'check-circle' },
    ];

    const currentStep = result?.current_step || 1;
    const currentRound = result?.current_round || 0;
    const status = data?.status || 'running';

    list.innerHTML = steps.map((s, idx) => {
        const stepNum = idx + 1;
        let state = 'pending';
        if (status === 'completed') {
            state = 'done';
        } else if (stepNum < currentStep) {
            state = 'done';
        } else if (stepNum === currentStep) {
            state = 'running';
        }

        const iconSvg = state === 'done'
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" width="12" height="12"><polyline points="20 6 9 17 4 12"/></svg>'
            : state === 'running'
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><circle cx="12" cy="12" r="10"/></svg>';

        const time = dashProcessSteps[stepNum]?.time || '';
        const name = (s.key === 'step4' && currentRound > 0)
            ? `${s.name}（第${currentRound}轮）`
            : s.name;

        return `
            <div class="dashboard-process-item">
                <div class="dashboard-process-icon ${state}">${iconSvg}</div>
                <div class="dashboard-process-name">${name}</div>
                <div class="dashboard-process-time">${time}</div>
            </div>
        `;
    }).join('');

    // 记录完成时间
    if (status === 'completed' || currentStep > 1) {
        for (let i = 1; i < currentStep; i++) {
            if (!dashProcessSteps[i]) {
                dashProcessSteps[i] = { time: new Date().toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }).replace(/\//g, '-') };
            }
        }
    }
}

function openLogModal() {
    const modal = document.getElementById('log-modal');
    if (!modal) return;

    // 渲染日志内容
    const logBody = document.getElementById('log-modal-body');
    if (logBody) {
        if (pipelineLogs.length === 0) {
            logBody.innerHTML = '<div class="log-empty">暂无日志记录，请先运行全流程。</div>';
        } else {
            logBody.innerHTML = pipelineLogs.map(entry => {
                const levelClass = entry.level === 'error' ? 'log-level-error'
                                 : entry.level === 'success' ? 'log-level-success'
                                 : entry.level === 'warn' ? 'log-level-warn'
                                 : 'log-level-info';
                const levelTag = entry.level === 'error' ? 'ERROR'
                               : entry.level === 'success' ? 'OK'
                               : entry.level === 'warn' ? 'WARN'
                               : 'INFO';
                return `<div class="log-entry ${levelClass}">
                    <span class="log-time">${entry.time}</span>
                    <span class="log-tag">${levelTag}</span>
                    <span class="log-msg">${_escapeHtml(entry.message)}</span>
                </div>`;
            }).join('');
            // 滚动到底部
            setTimeout(() => { logBody.scrollTop = logBody.scrollHeight; }, 50);
        }
    }

    modal.style.display = 'flex';
    // 更新日志条数
    const countEl = document.getElementById('log-modal-count');
    if (countEl) countEl.textContent = `${pipelineLogs.length} 条`;
}

function closeLogModal() {
    const modal = document.getElementById('log-modal');
    if (modal) modal.style.display = 'none';
}

function clearPipelineLogs() {
    pipelineLogs = [];
    const logBody = document.getElementById('log-modal-body');
    if (logBody) logBody.innerHTML = '<div class="log-empty">日志已清空。</div>';
    const countEl = document.getElementById('log-modal-count');
    if (countEl) countEl.textContent = '0 条';
}

/**
 * 追加一条日志
 * @param {'info'|'success'|'warn'|'error'} level
 * @param {string} message
 */
function _appendLog(level, message) {
    if (!message) return;
    const now = new Date();
    const time = now.toLocaleTimeString('zh-CN', { hour12: false });
    pipelineLogs.push({ time, level, message });
    // 实时更新（若模态框已打开）
    const modal = document.getElementById('log-modal');
    if (modal && modal.style.display !== 'none') {
        const logBody = document.getElementById('log-modal-body');
        if (logBody) {
            const levelClass = level === 'error' ? 'log-level-error'
                             : level === 'success' ? 'log-level-success'
                             : level === 'warn' ? 'log-level-warn'
                             : 'log-level-info';
            const levelTag = level === 'error' ? 'ERROR'
                           : level === 'success' ? 'OK'
                           : level === 'warn' ? 'WARN'
                           : 'INFO';
            const div = document.createElement('div');
            div.className = `log-entry ${levelClass}`;
            div.innerHTML = `<span class="log-time">${time}</span><span class="log-tag">${levelTag}</span><span class="log-msg">${_escapeHtml(message)}</span>`;
            logBody.appendChild(div);
            logBody.scrollTop = logBody.scrollHeight;
        }
        const countEl = document.getElementById('log-modal-count');
        if (countEl) countEl.textContent = `${pipelineLogs.length} 条`;
    }
}

function _escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ============================================
// 导出全局
// ============================================

window.runPipeline                = runPipeline;
window.pausePipeline              = pausePipeline;
window.resumePipeline             = resumePipeline;
window.restartPipeline            = restartPipeline;
window.addPipelineClass           = addPipelineClass;
window.addPipelineDeleteRange     = addPipelineDeleteRange;
window.setPipelineDeleteMode      = setPipelineDeleteMode;
window.loadReplacementPreview     = loadReplacementPreview;
window.initPipelineEventListeners = initPipelineEventListeners;
window.updateTargetFileDisplay    = updateTargetFileDisplay;
window.openLogModal               = openLogModal;
window.closeLogModal              = closeLogModal;
window.clearPipelineLogs          = clearPipelineLogs;
window.setExecMode                = setExecMode;
window.updateNumProcessesDisplay  = updateNumProcessesDisplay;
window.checkGPUStatus             = checkGPUStatus;
