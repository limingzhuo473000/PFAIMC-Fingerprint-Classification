/**
 * Config - 应用配置
 */

const AppConfig = {
    // API 配置
    api: {
        baseUrl: '/api',   // 相对路径，自动适配任意主机/端口
        timeout: 120000
    },

    // 地物类别颜色配置
    classColors: [
        '#16a34a', // 绿色
        '#2563eb', // 蓝色
        '#dc2626', // 红色
        '#ca8a04', // 黄色
        '#7c3aed', // 紫色
        '#0891b2', // 青色
        '#c026d3', // 粉色
        '#ea580c'  // 橙色
    ],

    // 类别颜色名称
    classColorNames: [
        '未分类',
        '边框',
        '建筑',
        '道路',
        '耕地',
        '林地',
        '草地',
        '水体',
        '裸地'
    ],

    // 默认类别
    defaultClasses: [
        { name: '建筑', values: [1, 2] },
        { name: '道路', values: [3] },
        { name: '耕地', values: [4, 5] },
        { name: '林地', values: [6] },
        { name: '草地', values: [7] },
        { name: '水体', values: [8] }
    ]
};

// 导出配置
window.AppConfig = AppConfig;
