#ifndef PRODUCT_DATABASE_HPP
#define PRODUCT_DATABASE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 商品元数据与特征结构体
struct ProductRecord {
    uint32_t tempId = 0;
    uint64_t id = 0;
    std::string name;
    std::string barcode;
    std::vector<float> feature; // 455维
    uint32_t matchCount = 0;

    bool operator==(const ProductRecord& other)
    {
        return id == other.id;
    }
};

// 检索结果结构体
struct SearchResult {
    uint64_t tempId;
    uint64_t id;
    std::string name;
    std::string barcode;
    float similarity;
    uint32_t recordIndex;
};

class ProductDatabase {
private:
    std::vector<ProductRecord> records;
    cv::Mat featureMatrix;
    std::vector<SearchResult> lastResults;
    uint32_t maxTempId = 0;

    // 重新构建内存矩阵
    void rebuildMatrix();

    void updateTemplateFeature(size_t targetIdx, const std::vector<float>& newFeature);

    void appendTemplate(const ProductRecord& newRecord);

    void removeTemplateAt(size_t removeIdx);

public:
    // 1. 添加商品到库中
    bool addProduct(const ProductRecord& record);

    // 2. 持久化：保存到二进制文件
    bool saveModel(const std::string& filepath);

    // 3. 加载：从二进制文件读取
    bool loadModel(const std::string& filepath);

    // 4. 高效检索 Top-K 最相似商品
    std::vector<SearchResult> search(const std::vector<float>& queryFeature, int topK = 3);
};

#endif // PRODUCT_DATABASE_HPP
