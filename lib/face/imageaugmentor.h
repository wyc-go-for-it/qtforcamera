#pragma once

#include "mylib/mylib_export.h"
#include <opencv2/opencv.hpp>
#include <vector>

class WYCAI_EXPORT ImageAugmentor {
public:
    /* ==================== 像素级增强 (不影响 BBox) ====================*/

    // 1. 调整亮度 (加法) delta > 0 变亮，< 0 变暗，推荐范围 [-50, 50]
    static cv::Mat adjustBrightness(const cv::Mat& src, int delta);

    // 2. 调整对比度 (乘法) alpha > 1 增强，< 1 减弱，推荐范围 [0.5, 1.5]
    static cv::Mat adjustContrast(const cv::Mat& src, float alpha);

    // 3. 同时调整亮度和对比度 (公式: dst = src * alpha + beta)
    static cv::Mat adjustBrightContrast(const cv::Mat& src, float alpha, int beta);

    // 4. Gamma 校正 (比线性亮度更自然) gamma < 1 变亮，> 1 变暗，推荐 [0.6, 1.5]
    static cv::Mat adjustGamma(const cv::Mat& src, double gamma);

    // 5. 调整 HSV (色相偏移, 饱和度缩放, 明度缩放)
    static cv::Mat adjustHSV(const cv::Mat& src, int hShift, float sScale, float vScale);

    // 6. 添加高斯噪声 (stddev 推荐 [5, 20])
    static cv::Mat addGaussianNoise(const cv::Mat& src, double mean, double stddev);

    // 7. 高斯模糊 (ksize 必须是奇数且 >= 3，推荐 3 或 5)
    static cv::Mat addBlur(const cv::Mat& src, int ksize);

    // ==================== 随机增强 (离线扩增数据集用) ====================

    // 随机应用上述像素级增强，返回增强后的图像
    static cv::Mat randomPixelAugment(const cv::Mat& src, cv::RNG& rng);

    /* ==================== 几何增强 (⚠️ 需同步修改 BBox 标签) ====================*/

    // 水平翻转 (返回翻转后的图，并同步修改传入的 BBox 列表)
    static cv::Mat flipHorizontal(cv::Mat& src, std::vector<cv::Rect>& bboxes);
    // 10. 随机旋转 (测试人脸对齐/姿态鲁棒性)
    static cv::Mat rotate(const cv::Mat& src, float angle);
    // 11. 随机缩放 (模拟远近)
    static cv::Mat scale(const cv::Mat& src, float factor);
};
