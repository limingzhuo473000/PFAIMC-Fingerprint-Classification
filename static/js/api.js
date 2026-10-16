/**
 * API - API 请求模块
 * 封装所有与后端的通信
 */

// ============================================
// API 基类
// ============================================

const API = {
    baseUrl: AppConfig.api.baseUrl,
    timeout: AppConfig.api.timeout,

    /**
     * 通用请求方法
     * @param {string} endpoint - API端点
     * @param {Object} options - 请求选项
     * @param {number|null} timeout - 超时毫秒数，null=不超时，未传=使用默认
     * @returns {Promise<Object>} 响应数据
     */
    async request(endpoint, options = {}, timeout = undefined) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        // timeout=null 表示不设超时（用于长时任务）
        const effectiveTimeout = timeout === undefined ? this.timeout : timeout;

        let controller = null;
        let timeoutId  = null;

        if (effectiveTimeout !== null && effectiveTimeout > 0) {
            controller = new AbortController();
            timeoutId  = setTimeout(() => controller.abort(), effectiveTimeout);
        }

        try {
            const fetchOptions = { ...config };
            if (controller) fetchOptions.signal = controller.signal;

            const response = await fetch(url, fetchOptions);

            if (timeoutId) clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                // FastAPI 返回 {detail: "..."}，Spring 等框架返回 {message: "..."}
                const errText = error.detail || error.message || JSON.stringify(error);
                throw new Error(errText || `请求失败: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (timeoutId) clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('请求超时');
            }
            throw error;
        }
    },

    /**
     * GET 请求
     * @param {string} endpoint - API端点
     * @returns {Promise<Object>}
     */
    get(endpoint, timeout = undefined) {
        return this.request(endpoint, { method: 'GET' }, timeout);
    },

    /**
     * POST 请求
     * @param {string} endpoint - API端点
     * @param {Object} data - 请求数据
     * @param {number|null} timeout - 超时毫秒数，null=不超时
     * @returns {Promise<Object>}
     */
    post(endpoint, data, timeout = undefined) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        }, timeout);
    }
};

// ============================================
// 任务 API
// ============================================

const TaskAPI = {
    /**
     * 获取任务状态
     * @param {string} taskId - 任务ID
     * @returns {Promise<Object>}
     */
    async getStatus(taskId) {
        return API.get(`/tasks/${taskId}`);
    },

    /**
     * 轮询任务状态
     * @param {string} taskId - 任务ID
     * @param {number} stepNum - 步骤编号
     * @param {Function} onUpdate - 更新回调
     * @param {Function} onComplete - 完成回调
     * @param {Function} onError - 错误回调
     */
    async pollStatus(taskId, stepNum, onUpdate, onComplete, onError) {
        const checkStatus = async () => {
            try {
                const data = await this.getStatus(taskId);

                if (data.status === 'running') {
                    onUpdate(data);
                    setTimeout(checkStatus, 1000);
                } else if (data.status === 'completed') {
                    onComplete(data);
                } else if (data.status === 'failed') {
                    onError(data);
                }
            } catch (error) {
                onError({ error: error.message });
            }
        };

        checkStatus();
    }
};

// ============================================
// 步骤 API
// ============================================

const StepsAPI = {
    /**
     * 运行步骤1
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep1(params) {
        return API.post('/step1/run', {
            input_img_dir: params.inputImgDir,
            output_dir: params.outputDir,
            filename: params.filename,
            num_classes: params.numClasses,
            delete_ranges: params.deleteRanges,
            adjust_sizes: params.adjustSizes,
            skip_discretization: !!params.skipDiscretization,  // 消融实验1：跳过离散化
            normalize_continuous: !!params.normalizeContinuous,  // 消融实验1配套：逐波段标准化
            zero_policy:          params.zeroPolicy || 'include'  // 分位数统计 0 值处理策略 include/exclude/auto（默认 include，与 92.66% 历史实验一致）
        });
    },

    /**
     * 运行步骤2
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep2(params) {
        return API.post('/step2/run', {
            shp_input: params.shpInput,
            tif_folder: params.tifFolder,
            experiment_data_path: params.experimentData,
            output_dir: params.outputDir
        });
    },

    /**
     * 运行步骤3
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep3(params) {
        return API.post('/step3/run', {
            images_folder: params.imagesFolder,
            samples_folder: params.samplesFolder,
            target_file: params.targetFile,
            output_file: params.outputFile,
            match_order: params.matchOrder,
            rounds: params.rounds,
            distance_metric: params.distanceMetric || 'hamming',  // 消融实验2：hamming/euclidean
            hamming_tolerance: params.hammingTolerance || 0       // 消融实验1配套：汉明容差 ε
        });
    },

    /**
     * 运行步骤4
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep4(params) {
        return API.post('/step4/run', {
            input_csv: params.inputCsv,
            images_folder: params.imagesFolder,
            output_dir: params.outputDir,
            add_geo: params.addGeo,
            max_sequence_length: params.maxSeqLength
        });
    },

    /**
     * 运行步骤5
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep5(params) {
        return API.post('/step5/run', {
            dat_path: params.datPath,
            classes: params.classes
        });
    },

    /**
     * 运行步骤6
     * @param {Object} params - 参数
     * @returns {Promise<Object>}
     */
    async runStep6(params) {
        return API.post('/step6/run', {
            reclass_dat: params.reclassDat,
            shp_folder: params.shpFolder,
            output_dir: params.outputDir
        });
    }
};

// 导出API对象
window.API = API;
window.TaskAPI = TaskAPI;
window.StepsAPI = StepsAPI;
