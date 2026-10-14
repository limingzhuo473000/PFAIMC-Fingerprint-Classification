/**
 * Navigation - 导航模块
 * 处理页面切换和折叠导航
 */

// ============================================
// 页面导航
// ============================================

/**
 * 导航到指定页面
 * @param {string} pageName - 页面名称
 */
function navigateToPage(pageName) {
    // 隐藏所有页面
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
    });

    // 取消所有导航项的激活状态
    document.querySelectorAll('.nav-item').forEach(n => {
        n.classList.remove('active');
    });

    // 显示目标页面
    const targetPage = document.getElementById('page-' + pageName);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // 激活对应的导航项
    const targetNav = document.querySelector(`.nav-item[data-page="${pageName}"]`);
    if (targetNav) {
        targetNav.classList.add('active');
    }

    // 展开对应的导航区域
    const navItem = targetNav || document.querySelector(`.nav-item[data-page="${pageName}"]`);
    if (navItem) {
        const section = navItem.closest('.nav-section-content');
        if (section && section.classList.contains('collapsed')) {
            section.classList.remove('collapsed');
            const header = section.previousElementSibling;
            if (header && header.classList.contains('nav-section-header')) {
                header.classList.remove('collapsed');
            }
        }
    }
}

/**
 * 显示页面（别名，兼容旧代码）
 * @param {string} page - 页面名称
 */
function showPage(page) {
    navigateToPage(page);
}

// ============================================
// 折叠导航
// ============================================

/**
 * 切换折叠区域
 * @param {string} sectionId - 区域ID
 * @param {HTMLElement} headerEl - 头部元素
 */
function toggleSection(sectionId, headerEl) {
    const section = document.getElementById(sectionId);
    if (!section) return;

    const isCollapsed = section.classList.toggle('collapsed');

    if (headerEl) {
        headerEl.classList.toggle('collapsed', isCollapsed);
    }
}

/**
 * 初始化折叠导航事件
 */
function initCollapsibleNav() {
    // 查找所有折叠区域头部
    document.querySelectorAll('.nav-section-header').forEach(header => {
        // 使用 data-target 属性指定要切换的区域
        const targetId = header.dataset.target;
        if (targetId) {
            header.addEventListener('click', () => {
                const target = document.getElementById(targetId);
                if (target) {
                    const isCollapsed = target.classList.toggle('collapsed');
                    header.classList.toggle('collapsed', isCollapsed);
                }
            });
        } else {
            // 兼容旧写法
            const toggleIcon = header.querySelector('.nav-section-toggle');
            if (toggleIcon) {
                header.addEventListener('click', () => {
                    const section = header.nextElementSibling;
                    if (section && section.classList.contains('nav-section-content')) {
                        const isCollapsed = section.classList.toggle('collapsed');
                        header.classList.toggle('collapsed', isCollapsed);
                    }
                });
            }
        }
    });
}

// ============================================
// 初始化
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // 初始化折叠导航
    initCollapsibleNav();

    // 绑定导航点击事件
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page) {
                navigateToPage(page);
            }
        });
    });
});
