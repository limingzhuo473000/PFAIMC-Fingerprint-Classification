/**
 * Forms - 表单处理模块
 * 处理文件输入、动态列表、表单切换等
 */

// ============================================
// 文件浏览
// ============================================

/**
 * 浏览文件夹
 * @param {string} inputId - 输入框ID
 */
async function browseFolder(inputId) {
    const input = document.getElementById(inputId);
    const path = prompt('请输入文件夹路径:', input?.value?.trim() || 'D:/');
    if (path) {
        input.value = path;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

/**
 * 浏览文件
 * @param {string} inputId - 输入框ID
 * @param {boolean} save - 是否为保存文件
 */
async function browseFile(inputId, save = false) {
    const input = document.getElementById(inputId);
    const path = prompt(save ? '请输入输出文件路径:' : '请输入文件路径:', input?.value?.trim() || 'D:/');
    if (path) {
        input.value = path;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

// ============================================
// 切换功能
// ============================================

/**
 * 切换删除波段模式 (Step 1)
 * @param {string} mode - 'default' 或 'custom'
 */
function setDeleteMode(mode) {
    document.querySelectorAll('#step1-delete-mode .radio-option').forEach(el => {
        el.classList.toggle('active', el.dataset.value === mode);
    });

    const wrapper = document.getElementById('step1-delete-ranges-wrapper');
    const note = document.getElementById('step1-delete-default-note');

    if (wrapper) {
        wrapper.style.display = mode === 'custom' ? 'block' : 'none';
    }
    if (note) {
        note.style.display = mode === 'default' ? 'block' : 'none';
    }
}

/**
 * 切换匹配模式 (Step 3)
 * @param {string} mode - 'default' 或 'custom'
 */
function setMatchMode(mode) {
    document.querySelectorAll('#step3-match-mode .radio-option').forEach(el => {
        el.classList.toggle('active', el.dataset.value === mode);
    });

    const defaultDiv = document.getElementById('step3-match-default');
    const customDiv = document.getElementById('step3-match-custom');

    if (defaultDiv) {
        defaultDiv.style.display = mode === 'default' ? 'block' : 'none';
    }
    if (customDiv) {
        customDiv.style.display = mode === 'custom' ? 'block' : 'none';
    }
}

/**
 * 切换距离度量 (Step 3，消融实验2)
 * @param {string} metric - 'hamming' 或 'euclidean'
 */
function setStep3DistanceMetric(metric) {
    document.querySelectorAll('#step3-distance-metric .radio-option').forEach(el => {
        el.classList.toggle('active', el.dataset.value === metric);
    });
}

// ============================================
// 动态列表
// ============================================

// 计数器
let customMatchCounter = 0;
let roundCounter = 0;
let classCounter = 0;

/**
 * 添加波段范围 (Step 1)
 */
function addDeleteRange() {
    const container = document.getElementById('step1-delete-ranges');
    if (!container) return;

    const div = document.createElement('div');
    div.className = 'dynamic-item';
    div.innerHTML = `
        <input type="number" class="form-input" placeholder="起始" min="1" style="width: 80px;">
        <span style="color: var(--text-muted);">~</span>
        <input type="number" class="form-input" placeholder="结束" min="1" style="width: 80px;">
        <button class="btn-remove" onclick="this.parentElement.remove()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;
    container.appendChild(div);
}

/**
 * 添加自定义匹配项 (Step 3)
 */
function addCustomMatchItem() {
    const container = document.getElementById('step3-custom-match-list');
    if (!container) return;

    customMatchCounter++;
    const div = document.createElement('div');
    div.className = 'dynamic-item';
    div.innerHTML = `
        <span style="color: var(--text-muted); font-family: 'Geist Mono', monospace; font-size: 0.8125rem; min-width: 24px;">${customMatchCounter}.</span>
        <input type="text" class="form-input" placeholder="类别名称 (如 建筑)" style="flex: 1;">
        <button class="btn-remove" onclick="reindexCustomMatches(); this.parentElement.remove();">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;
    container.appendChild(div);
}

/**
 * 重新索引自定义匹配项
 */
function reindexCustomMatches() {
    const container = document.getElementById('step3-custom-match-list');
    if (!container) return;

    const items = container.querySelectorAll('.dynamic-item');
    items.forEach((item, index) => {
        const span = item.querySelector('span');
        if (span) {
            span.textContent = (index + 1) + '.';
        }
    });
    customMatchCounter = items.length;
}

/**
 * 添加匹配轮次 (Step 3)
 */
function addRound() {
    const container = document.getElementById('step3-rounds');
    if (!container) return;

    roundCounter++;
    const div = document.createElement('div');
    div.className = 'dynamic-item';
    div.style.flexDirection = 'column';
    div.style.alignItems = 'stretch';
    div.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <strong>第 ${roundCounter} 轮</strong>
            <button class="btn-remove" onclick="this.parentElement.parentElement.remove()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </div>
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
            <input type="number" class="form-input round-threshold" placeholder="距离阈值（欧氏可设小数）" min="0" step="any" style="width: 160px;">
        </div>
        <div class="round-replacements"></div>
        <button class="btn-add" style="font-size: 0.75rem; padding: 0.25rem 0.5rem; margin-top: 0.5rem;" onclick="addReplacement(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            添加样本替换值
        </button>
    `;
    container.appendChild(div);
}

/**
 * 添加样本替换值
 * @param {HTMLElement} btn - 按钮元素
 */
function addReplacement(btn) {
    const container = btn.previousElementSibling;
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.gap = '0.5rem';
    div.style.marginBottom = '0.25rem';
    div.innerHTML = `
        <input type="text" class="form-input replacement-sample" placeholder="样本文件名" style="flex: 1;">
        <input type="text" class="form-input replacement-value" placeholder="替换值" style="width: 60px;">
        <button class="btn-remove" onclick="this.parentElement.remove()" style="width: 24px; height: 24px;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;
    container.appendChild(div);
}

/**
 * 添加地物类别 (Step 5)
 */
function addLandCoverClass() {
    const container = document.getElementById('step5-classes');
    if (!container) return;

    const idx = classCounter++;
    const color = AppConfig.classColors[idx % AppConfig.classColors.length];
    const div = document.createElement('div');
    div.className = 'dynamic-item';
    div.innerHTML = `
        <div style="width: 24px; height: 24px; background: ${color}; border-radius: 4px;"></div>
        <input type="text" class="form-input class-name" placeholder="类别名称" style="flex: 1;">
        <input type="text" class="form-input class-values" placeholder="数值 (如 1,2,7)" style="width: 150px;">
        <button class="btn-remove" onclick="this.parentElement.remove(); updateColorPreview();">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;
    container.appendChild(div);
    updateColorPreview();
}

/**
 * 更新颜色预览
 */
function updateColorPreview() {
    const container = document.getElementById('step5-color-preview');
    if (!container) return;

    const items = document.querySelectorAll('#step5-classes .dynamic-item');
    let html = `
        <div class="color-swatch" style="background: #808080;" title="未分类"></div>
        <div class="color-swatch" style="background: #000000;" title="边框"></div>
    `;

    items.forEach((item, i) => {
        const color = AppConfig.classColors[i % AppConfig.classColors.length];
        const name = item.querySelector('.class-name').value || '类别' + (i + 1);
        html += `<div class="color-swatch" style="background: ${color};" title="${name}"></div>`;
    });

    container.innerHTML = html;
}
