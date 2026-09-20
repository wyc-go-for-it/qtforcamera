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
    static const int featureDim = 448;

    /**
     * @brief 1. 背景差分并自动裁剪商品 ROI 区域
     */
    static std::pair<cv::Mat, cv::Mat> cropROI(const cv::Mat& bg, const cv::Mat& fg, double minArea = 1500.0, bool rotation = true);

    /**
     * @brief 2. 提取 2x2 空间网格 HSV 颜色直方图 (128维)
     */
    static std::vector<float> extractColorFeature(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat());

    /**
     * @brief 3. 提取 形状不变矩 (7维)
     */
    static std::vector<float> extractShapeFeature(const cv::Mat& roi, const cv::Mat& mask = cv::Mat());

    /**
     * @brief 4. 提取 LBP 局部二值模式纹理直方图 (256维)
     */
    static std::vector<float> extractTextureFeature(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat());

    /**
     * @brief 5. 组合主函数：提取 391 维复合归一化特征向量
     */
    static std::vector<float> extract(const cv::Mat& roi, const cv::Mat& croppedMask = cv::Mat(), FeatureWeights weights = FeatureWeights());

    /**
     * @brief 当用户二次选择确认商品 A 时，在线强化数据库中的特征向量
     * @param dbVec 数据库中商品 A 原有的特征向量 (455D)
     * @param queryVec 当前提取的查询向量 (455D)
     * @param alpha 旧特征的保留权重 (如 0.9)
     * @param beta 新特征的增强系数 (如 0.1)
     */
    static void reinforceFeatureVector(std::vector<float>& dbVec, const std::vector<float>& queryVec, float alpha = 0.9f, float beta = 0.1f);

    /**
     * @brief 6. 计算两个归一化向量的余弦相似度 [-1.0, 1.0]
     */
    static float computeSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB);

private:
    /**
     * @brief 3. 提取 Hu 形状不变矩 (7维)
     */
    static std::vector<float> extractHuShapeFeature(const cv::Mat& roi);

    /**
     * @brief 提取基于傅里叶描述子的形状特征（具备平移、旋转、缩放不变性）
     *
     * @param roi 输入图像 (BGR 或 灰度图)
     * @param mask 掩码图像 (可选，可传 cv::Mat()；若传则辅助准确提取轮廓)
     * @param numCoeffs 期望截取的低频描述子数量（决定返回向量的维度，默认 64 维）
     * @return std::vector<float> 归一化后的形状特征向量 (维度为 numCoeffs)
     */
    static std::vector<float> extractFourierShapeFeature(const cv::Mat& roi, const cv::Mat& mask = cv::Mat(), int numCoeffs = 64);

    static void normalizeVector(std::vector<float>& vec);
};

#endif // PRODUCT_FEATURE_ENGINE_HPP
