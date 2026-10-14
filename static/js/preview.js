/**
 * preview.js - 预览与格式转换面板
 *
 * 功能：
 *  - 浏览步骤输出目录中的CSV文件
 *  - 预览文件内容（前N行）
 *  - CSV → JSON / TXT 格式转换并下载
 */

// ── 面板显示/隐藏 ──
function togglePreviewPanel() {
    const panel = document.getElementById('preview-panel');
    if (!panel) return;
    const isHidden = panel.classList.contains('preview-panel-hidden');
    if (isHidden) {
        panel.classList.remove('preview-panel-hidden');
    } else {
        panel.classList.add('preview-panel-hidden');
    }
}

// ── 当前选中的文件 ──
let _previewCurrentFile = null;

// ── 浏览目录 ──
function quickBrowse(dirPath) {
    document.getElementById('preview-dir-input').value = dirPath;
    browseDir();
}

async function browseDir() {
    const dirInput = document.getElementById('preview-dir-input');
    const dirPath = dirInput.value.trim();
    if (!dirPath) return;

    const fileList = document.getElementById('preview-file-list');
    fileList.innerHTML = '<span style="color:var(--text-secondary);">加载中...</span>';

    try {
        const url = `/api/preview/list?dir_path=${encodeURIComponent(dirPath)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(await resp.text());

        const data = await resp.json();
        const files = data.files || [];

        if (files.length === 0) {
            fileList.innerHTML = '<span style="color:var(--text-secondary);">目录为空</span>';
            return;
        }

        fileList.innerHTML = files.map(f => {
            const icon = f.is_dir ? '📁' : '📄';
            const sizeStr = f.size > 1024
                ? `${(f.size / 1024 / 1024).toFixed(2)} MB`
                : `${(f.size / 1024).toFixed(1)} KB`;
            const extClass = f.ext === '.csv' ? '' : 'style="opacity:0.5"';
            return `<div class="file-item" ${extClass} onclick="previewFile('${f.path.replace(/\\/g, '\\\\')}', '${f.name}')">
                <span>${icon} ${f.name}</span>
                <span class="size">${f.is_dir ? '目录' : sizeStr}</span>
            </div>`;
        }).join('');

        // 高亮已选文件
        if (_previewCurrentFile) highlightSelected();
    } catch (err) {
        fileList.innerHTML = `<span style="color:var(--danger-color);">加载失败: ${err.message}</span>`;
    }
}

// ── 预览文件 ──
async function previewFile(filePath, fileName) {
    _previewCurrentFile = filePath;
    document.getElementById('preview-current-file').textContent = fileName || filePath;

    const contentDiv = document.getElementById('preview-content');
    contentDiv.innerHTML = '<span style="color:var(--text-secondary);">加载中...</span>';
    highlightSelected();

    try {
        const rows = document.getElementById('preview-rows-select').value || '50';
        const url = `/api/preview/csv?path=${encodeURIComponent(filePath)}&rows=${rows}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(await resp.text());

        const data = await resp.json();

        // 构建表格展示
        let html = `<div style="font-size:11px;color:var(--text-secondary);margin-bottom:6px;">
            文件: ${data.file} | 总行数: ${data.total_rows} | 预览: ${data.preview_rows}行 | 大小: ${(data.size/1024).toFixed(1)}KB
        </div>`;

        if (data.headers.length === 0) {
            html += '<div style="color:var(--text-secondary);">文件为空</div>';
        } else {
            html += '<table style="border-collapse:collapse;font-size:12px;width:max-content;min-width:100%;">';

            // 表头
            html += '<thead><tr style="background:var(--bg-secondary);">';
            html += '<th style="padding:4px 10px;border:1px solid var(--border-color);text-align:left;">#</th>';
            for (const h of data.headers) {
                html += `<th style="padding:4px 10px;border:1px solid var(--border-color);text-align:left;white-space:nowrap;">${h || '&nbsp;'}</th>`;
            }
            html += '</tr></thead><tbody>';

            // 数据行
            for (let i = 0; i < data.rows.length; i++) {
                const row = data.rows[i];
                html += '<tr>';
                html += `<td style="padding:3px 10px;border:1px solid var(--border-color);color:var(--text-secondary);">${i+1}</td>`;
                for (const cell of row) {
                    const display = cell && cell.length > 60 ? cell.substring(0, 60) + '...' : (cell || '');
                    html += `<td style="padding:3px 10px;border:1px solid var(--border-color);max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${cell || ''}">${display || '&nbsp;'}</td>`;
                }
                html += '</tr>';
            }

            html += '</tbody></table>';
        }

        contentDiv.innerHTML = html;
    } catch (err) {
        contentDiv.innerHTML = `<span style="color:var(--danger-color);">预览失败: ${err.message}</span>`;
    }
}

function refreshPreview() {
    if (_previewCurrentFile) {
        const fileName = _previewCurrentFile.split(/[\\/]/).pop();
        previewFile(_previewCurrentFile, fileName);
    }
}

function highlightSelected() {
    document.querySelectorAll('#preview-file-list .file-item').forEach(el => {
        el.classList.remove('selected');
    });
}

// ── 格式转换 ──
async function convertAndDownload() {
    if (!_previewCurrentFile) {
        alert('请先在文件列表中选择一个CSV文件');
        return;
    }

    const fmt = document.getElementById('convert-format-select').value;
    const resultDiv = document.getElementById('convert-result');

    try {
        const url = `/api/preview/convert?path=${encodeURIComponent(_previewCurrentFile)}&fmt=${fmt}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(await resp.text());

        const data = await resp.json();

        if (fmt === 'json') {
            // JSON: 创建Blob并下载
            const jsonStr = JSON.stringify(data.data, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const downloadUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = data.file;
            a.click();
            URL.revokeObjectURL(downloadUrl);
            resultDiv.textContent = `已下载: ${data.file} (${data.row_count}行)`;
            resultDiv.style.color = 'var(--success-color, #22c55e)';
        } else if (fmt === 'txt' || fmt === 'csv') {
            // TXT/CSV: 通过下载端点
            const dlUrl = `/api/preview/download?path=${encodeURIComponent(_previewCurrentFile)}&fmt=${fmt}`;
            const a = document.createElement('a');
            a.href = dlUrl;
            a.download = data.file;
            a.click();
            resultDiv.textContent = `已下载: ${data.file} (${data.row_count}行)`;
            resultDiv.style.color = 'var(--success-color, #22c55e)';
        }
    } catch (err) {
        resultDiv.textContent = `转换失败: ${err.message}`;
        resultDiv.style.color = 'var(--danger-color, #ef4444)';
    }
}

// 导出到全局
window.togglePreviewPanel = togglePreviewPanel;
window.browseDir = browseDir;
window.quickBrowse = quickBrowse;
window.previewFile = previewFile;
window.refreshPreview = refreshPreview;
window.convertAndDownload = convertAndDownload;
