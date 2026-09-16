#ifndef PRODUCT_DATABASE_HPP
#define PRODUCT_DATABASE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 商品元数据与特征结构体
struct ProductRecord {
    uint64_t id;
    std::string name;
    std::string barcode;
    std::vector<float> feature; // 391维
};

// 检索结果结构体
struct SearchResult {
    uint64_t id;
    std::string name;
    std::string barcode;
    float similarity;
};

class ProductDatabase {
private:
    std::vector<ProductRecord> records;
    cv::Mat featureMatrix; // 将所有向量压入 N x 391 的矩阵中加速计算
    const int featureDim = 391;

    // 重新构建内存矩阵
    void rebuildMatrix();

public:
    // 1. 添加商品到库中
    void addProduct(const ProductRecord& record);

    // 2. 持久化：保存到二进制文件
    bool saveToFile(const std::string& filepath);

    // 3. 加载：从二进制文件读取
    bool loadFromFile(const std::string& filepath);

    // 4. 高效检索 Top-K 最相似商品
    std::vector<SearchResult> search(const std::vector<float>& queryFeature, int topK = 3);
};

#endif // PRODUCT_DATABASE_HPP
