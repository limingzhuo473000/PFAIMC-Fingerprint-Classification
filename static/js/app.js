/**
 * App - 主应用入口
 * 初始化应用程序
 */

// ============================================
// 初始化
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // 初始化颜色预览
    updateColorPreview();

    // 添加初始地物类别
    addLandCoverClass();

    // 初始化全流程页面事件监听
    if (typeof initPipelineEventListeners === 'function') {
        initPipelineEventListeners();
    }

    // 检测GPU状态（异步，不阻塞页面加载）
    if (typeof checkGPUStatus === 'function') {
        setTimeout(() => checkGPUStatus(), 500);
    }

    console.log('遥感影像分类系统已初始化');
});

// ============================================
// 导出全局函数（供HTML onclick调用）
// ============================================

// 导航
window.navigateToPage = navigateToPage;
window.showPage = showPage;
window.toggleSection = toggleSection;

// 文件浏览
window.browseFolder = browseFolder;
window.browseFile = browseFile;

// 切换功能
window.setDeleteMode = setDeleteMode;
window.setMatchMode = setMatchMode;

// 步骤1 CSV 来源选择
window.setStep1CsvMode = setStep1CsvMode;
window.scanStep1CsvFiles = scanStep1CsvFiles;
window.selectStep1CsvFile = selectStep1CsvFile;
window.onExistingStep1CsvChange = onExistingStep1CsvChange;

// 动态列表
window.addDeleteRange = addDeleteRange;
window.addCustomMatchItem = addCustomMatchItem;
window.reindexCustomMatches = reindexCustomMatches;
window.addRound = addRound;
window.addReplacement = addReplacement;
window.addLandCoverClass = addLandCoverClass;
window.updateColorPreview = updateColorPreview;

// 步骤执行
window.runStep1 = runStep1;
window.runStep2 = runStep2;
window.runStep3 = runStep3;
window.runStep4 = runStep4;
window.runStep5 = runStep5;
window.runStep6 = runStep6;

// 步骤2相关
window.loadStep2Recommendations = loadStep2Recommendations;
window.handleStepComplete = handleStepComplete;

// 状态显示
window.showStatus = showStatus;
window.showTaskMonitor = showTaskMonitor;
window.hideTaskMonitor = hideTaskMonitor;
window.resetTaskMonitorTiming = resetTaskMonitorTiming;
window._formatDuration = _formatDuration;
