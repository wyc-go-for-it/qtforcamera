#ifndef PRODUCT_FEATURE_ENGINE_HPP
#define PRODUCT_FEATURE_ENGINE_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct FeatureWeights {
    float color = 1.0f; // 颜色权重
    float shape = 0.5f; // 形状权重
    float texture = 0.8f; // 纹理权重
};

class ProductFeatureEngine {
public:
    /**
     * @brief 1. 背景差分并自动裁剪商品 ROI 区域
     */
    static std::pair<cv::Mat, cv::Mat> cropROI(const cv::Mat& bg, const cv::Mat& fg, double minArea = 1500.0);

    /**
     * @brief 2. 提取 2x2 空间网格 HSV 颜色直方图 (128维)
     */
    static std::vector<float> extractColorFeature(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat());

    /**
     * @brief 3. 提取 Hu 形状不变矩 (7维)
     */
    static std::vector<float> extractShapeFeature(const cv::Mat& roi);

    /**
     * @brief 4. 提取 LBP 局部二值模式纹理直方图 (256维)
     */
    static std::vector<float> extractTextureFeature(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat());

    /**
     * @brief 5. 组合主函数：提取 391 维复合归一化特征向量
     */
    static std::vector<float> extract(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat(), FeatureWeights weights = FeatureWeights());

    /**
     * @brief 6. 计算两个归一化向量的余弦相似度 [-1.0, 1.0]
     */
    static float computeSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB);
};

#endif // PRODUCT_FEATURE_ENGINE_HPP
