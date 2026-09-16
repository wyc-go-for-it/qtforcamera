#include "productdatabase.h"
#include <algorithm>
#include <fstream>
#include <iostream>

void ProductDatabase::rebuildMatrix()
{
    if (records.empty()) {
        featureMatrix = cv::Mat();
        return;
    }
    featureMatrix = cv::Mat(static_cast<int>(records.size()), featureDim, CV_32F);
    for (size_t i = 0; i < records.size(); ++i) {
        float* rowPtr = featureMatrix.ptr<float>(static_cast<int>(i));
        std::copy(records[i].feature.begin(), records[i].feature.end(), rowPtr);
    }
}

void ProductDatabase::addProduct(const ProductRecord& record)
{
    if (record.feature.size() != static_cast<size_t>(featureDim)) {
        std::cerr << "特征维度不匹配！" << std::endl;
        return;
    }
    records.push_back(record);
    rebuildMatrix();
}

bool ProductDatabase::saveToFile(const std::string& filepath)
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open())
        return false;

    uint32_t count = static_cast<uint32_t>(records.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& item : records) {
        ofs.write(reinterpret_cast<const char*>(&item.id), sizeof(item.id));

        // 写入变长字符串 (Name)
        uint32_t nameLen = static_cast<uint32_t>(item.name.size());
        ofs.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        ofs.write(item.name.data(), nameLen);

        // 写入变长字符串 (Barcode)
        uint32_t codeLen = static_cast<uint32_t>(item.barcode.size());
        ofs.write(reinterpret_cast<const char*>(&codeLen), sizeof(codeLen));
        ofs.write(item.barcode.data(), codeLen);

        // 写入 391 维 float 特征数组
        ofs.write(reinterpret_cast<const char*>(item.feature.data()), featureDim * sizeof(float));
    }
    return true;
}

bool ProductDatabase::loadFromFile(const std::string& filepath)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
        return false;

    records.clear();
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (uint32_t i = 0; i < count; ++i) {
        ProductRecord item;
        ifs.read(reinterpret_cast<char*>(&item.id), sizeof(item.id));

        uint32_t nameLen = 0;
        ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        item.name.resize(nameLen);
        ifs.read(&item.name[0], nameLen);

        uint32_t codeLen = 0;
        ifs.read(reinterpret_cast<char*>(&codeLen), sizeof(codeLen));
        item.barcode.resize(codeLen);
        ifs.read(&item.barcode[0], codeLen);

        item.feature.resize(featureDim);
        ifs.read(reinterpret_cast<char*>(item.feature.data()), featureDim * sizeof(float));

        records.push_back(item);
    }

    rebuildMatrix();
    return true;
}

std::vector<SearchResult> ProductDatabase::search(const std::vector<float>& queryFeature, int topK)
{
    if (featureMatrix.empty() || queryFeature.size() != static_cast<size_t>(featureDim)) {
        return {};
    }

    // 将 query 转为 391 x 1 列向量
    cv::Mat queryMat(featureDim, 1, CV_32F, const_cast<float*>(queryFeature.data()));

    // 矩阵乘法： (N x 391) * (391 x 1) = (N x 1) 得分矩阵
    cv::Mat scores = featureMatrix * queryMat;

    // 提取得分并按相似度降序排序
    std::vector<std::pair<float, size_t>> scoreIndexMap;
    scoreIndexMap.reserve(records.size());
    for (int i = 0; i < scores.rows; ++i) {
        scoreIndexMap.push_back({ scores.at<float>(i, 0), static_cast<size_t>(i) });
    }

    std::partial_sort(scoreIndexMap.begin(),
        scoreIndexMap.begin() + std::min<size_t>(topK, scoreIndexMap.size()),
        scoreIndexMap.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // 组装返回结果
    std::vector<SearchResult> results;
    for (size_t i = 0; i < std::min<size_t>(topK, scoreIndexMap.size()); ++i) {
        size_t idx = scoreIndexMap[i].second;
        results.push_back({ records[idx].id,
            records[idx].name,
            records[idx].barcode,
            scoreIndexMap[i].first });
    }
    return results;
}
