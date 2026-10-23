#include "productdatabase.h"
#include "productfeatureengine.h"
#include <algorithm>
#include <fstream>
#include <iostream>

// vec_sim.h
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace vecsim {
/**
 * @brief reduceCosToTarget 把 cos(A,B) 精确降到 target_cos，A 完全不动，B 改动后重新归一化,要求：A、B 都是 L2 归一化的单位向量，target_cos ∈ (-1, 1)
 * @param A
 * @param B
 * @param target_cos ∈ (-1, 1)
 * @return
 */
inline std::vector<float> reduceCosToTarget(const std::vector<float>& A,
    const std::vector<float>& B,
    double target_cos)
{
    if (A.size() != B.size() || target_cos <= -1.0 || target_cos >= 1.0) {
        return B;
    }

    double b_par = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        b_par += double(A[i]) * double(B[i]);

    if (b_par <= target_cos)
        return B;

    // b_perp² = 1 - b_par²（B 是单位向量）
    const double b_perp_sq = std::max(1.0 - b_par * b_par, 0.0);
    if (b_perp_sq < 1e-12) {
        // A、B 共线，无法降到中间值，保守返回原 B
        return B;
    }
    const double b_perp = std::sqrt(b_perp_sq);

    // b_par' = c * b_perp / sqrt(1 - c²)
    const double c2 = target_cos * target_cos;
    const double b_par_target = target_cos * b_perp / std::sqrt(1.0 - c2);

    // t = 1 - b_par'/b_par
    const double t = 1.0 - b_par_target / b_par;

    // B' = B - t * b_par * A
    std::vector<float> B_new(B.size());
    for (size_t i = 0; i < B.size(); ++i) {
        const double v = double(B[i]) - t * b_par * double(A[i]);
        B_new[i] = float(v);
    }

    // 重新 L2 归一化（改动后模长变了，必须归一化回单位向量）
    double n = 0.0;
    for (float x : B_new)
        n += double(x) * double(x);
    n = std::sqrt(n);
    if (n > 1e-12) {
        const float inv = float(1.0 / n);
        for (float& x : B_new)
            x *= inv;
    }
    return B_new;
}

inline std::vector<float> makeOrthogonal(const std::vector<float>& A,
    const std::vector<float>& B)
{
    if (A.size() != B.size()) {
        return B;
    }

    double b_par = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        b_par += double(A[i]) * double(B[i]);

    std::vector<float> B_new(B.size());
    for (size_t i = 0; i < B.size(); ++i)
        B_new[i] = float(double(B[i]) - b_par * double(A[i]));

    double n = 0.0;
    for (float x : B_new)
        n += double(x) * double(x);
    n = std::sqrt(n);
    if (n > 1e-12) {
        const float inv = float(1.0 / n);
        for (float& x : B_new)
            x *= inv;
    }
    return B_new;
}

/**
 * @brief reduceCosByFactor 按因子 t 直接削减余弦相似度,t=1 即正交
 * @param A
 * @param B
 * @param t
 * @return
 */
inline std::vector<float> reduceCosByFactor(const std::vector<float>& A,
    const std::vector<float>& B,
    double t)
{
    if (A.size() != B.size()) {
        return B;
    }

    double b_par = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        b_par += double(A[i]) * double(B[i]);

    std::vector<float> B_new(B.size());
    for (size_t i = 0; i < B.size(); ++i)
        B_new[i] = float(double(B[i]) - t * b_par * double(A[i]));

    double n = 0.0;
    for (float x : B_new)
        n += double(x) * double(x);
    n = std::sqrt(n);
    if (n > 1e-12) {
        const float inv = float(1.0 / n);
        for (float& x : B_new)
            x *= inv;
    }
    return B_new;
}

} // namespace vecsim

void ProductDatabase::rebuildMatrix()
{
    if (records.empty()) {
        featureMatrix = cv::Mat();
        return;
    }
    featureMatrix = cv::Mat(static_cast<int>(records.size()), ProductFeatureEngine::featureDim, CV_32F);
    for (size_t i = 0; i < records.size(); ++i) {
        float* rowPtr = featureMatrix.ptr<float>(static_cast<int>(i));
        std::copy(records[i].feature.begin(), records[i].feature.end(), rowPtr);
    }
}

bool ProductDatabase::addProduct(const ProductRecord& record)
{
    static const int MAX_TEMPLATES = 5; // 每个商品最多保留 5 个姿态模板
    static const float MERGE_THRESH = 0.85f; // 判定为相同姿态的相似度阈值

    if (record.feature.size() != static_cast<size_t>(ProductFeatureEngine::featureDim)) {
        return false;
    }
    ProductRecord newRecord = record;

    if (!lastResults.empty() && newRecord.tempId > 0) { // 校准商品
        for (const auto& result : lastResults) {
            if (result.id != newRecord.id) {
                if (result.similarity > MERGE_THRESH && result.recordIndex <= records.size() - 1) {
                    auto& existFeature = records.at(result.recordIndex).feature;

                    existFeature = vecsim::reduceCosByFactor(newRecord.feature, existFeature, 0.3);

                    updateTemplateFeature(result.recordIndex, existFeature);
                }
            }
        }
    }

    std::vector<size_t> matchedIndices;

    const auto id = newRecord.id;
    for (size_t i = 0, size = records.size(); i < size; ++i) {
        if (records.at(i).id == id) {
            matchedIndices.push_back(i);
        }
    }

    std::cout << "tempId:" << newRecord.tempId << " matchedIndices:" << matchedIndices.size() << std::endl;

    if (newRecord.tempId == 0 && matchedIndices.size() < MAX_TEMPLATES) {
        newRecord.tempId = (++maxTempId);
        appendTemplate(newRecord);
    } else {

        int bestIdx = -1;
        float maxScore = -1.0f;

        for (size_t i = 0; i < matchedIndices.size(); ++i) {
            size_t idx = matchedIndices.at(i);
            float score = ProductFeatureEngine::computeSimilarity(records.at(idx).feature, newRecord.feature);
            if (score > maxScore) {
                maxScore = score;
                bestIdx = idx;
            }
        }

        if (bestIdx != -1 && maxScore >= MERGE_THRESH) {
            records.at(bestIdx).matchCount++;

            auto& existFeature = records.at(bestIdx).feature;
            ProductFeatureEngine::reinforceFeatureVector(existFeature, newRecord.feature);
            updateTemplateFeature(bestIdx, existFeature);

            std::cout << "bestIdx:" << bestIdx << " maxScore:" << maxScore << " matchCount:" << records.at(bestIdx).matchCount << std::endl;
        } else {
            if (matchedIndices.size() >= MAX_TEMPLATES) {
                int targetEliminateIdx = -1;
                uint32_t minCount = UINT32_MAX;

                for (size_t idx : matchedIndices) {
                    if (records.at(idx).matchCount <= minCount) {
                        minCount = records.at(idx).matchCount;
                        targetEliminateIdx = static_cast<int>(idx);
                    }
                }

                std::cout << "targetEliminateIdx:" << targetEliminateIdx << std::endl;

                if (targetEliminateIdx != -1) {
                    if (records.at(targetEliminateIdx).matchCount == 0) {
                        ProductFeatureEngine::reinforceFeatureVector(newRecord.feature, records.at(targetEliminateIdx).feature);
                    }
                    newRecord.tempId = records.at(targetEliminateIdx).tempId;
                    removeTemplateAt(targetEliminateIdx);
                }
            }

            appendTemplate(newRecord);
        }
    }
    return true;
}

void ProductDatabase::updateTemplateFeature(size_t targetIdx, const std::vector<float>& newFeature)
{
    if (targetIdx >= records.size())
        return;

    records[targetIdx].feature = newFeature;

    float* rowPtr = featureMatrix.ptr<float>(static_cast<int>(targetIdx));
    std::copy(newFeature.begin(), newFeature.end(), rowPtr);
}

void ProductDatabase::appendTemplate(const ProductRecord& newRecord)
{
    records.push_back(newRecord);

    cv::Mat rowMat(1, ProductFeatureEngine::featureDim, CV_32F, const_cast<float*>(newRecord.feature.data()));

    if (featureMatrix.empty()) {
        featureMatrix = rowMat.clone();
    } else {
        featureMatrix.push_back(rowMat);
    }
}

void ProductDatabase::removeTemplateAt(size_t removeIdx)
{
    if (removeIdx >= records.size())
        return;

    size_t lastIdx = records.size() - 1;

    if (removeIdx != lastIdx) {
        records[removeIdx] = std::move(records[lastIdx]);

        cv::Mat lastRow = featureMatrix.row(static_cast<int>(lastIdx));
        cv::Mat targetRow = featureMatrix.row(static_cast<int>(removeIdx));

        lastRow.copyTo(targetRow);
    }

    records.pop_back();

    featureMatrix = featureMatrix.rowRange(0, static_cast<int>(records.size()));
}

bool ProductDatabase::saveModel(const std::string& filepath)
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open())
        return false;

    uint32_t count = static_cast<uint32_t>(records.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& item : records) {
        ofs.write(reinterpret_cast<const char*>(&item.tempId), sizeof(item.tempId));

        ofs.write(reinterpret_cast<const char*>(&item.id), sizeof(item.id));

        // 写入变长字符串 (Name)
        uint32_t nameLen = static_cast<uint32_t>(item.name.size());
        ofs.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        ofs.write(item.name.data(), nameLen);

        // 写入变长字符串 (Barcode)
        uint32_t codeLen = static_cast<uint32_t>(item.barcode.size());
        ofs.write(reinterpret_cast<const char*>(&codeLen), sizeof(codeLen));
        ofs.write(item.barcode.data(), codeLen);

        // 写入 featureDim 维 float 特征数组
        ofs.write(reinterpret_cast<const char*>(item.feature.data()), ProductFeatureEngine::featureDim * sizeof(float));

        ofs.write(reinterpret_cast<const char*>(&item.matchCount), sizeof(item.matchCount));
    }
    return true;
}

bool ProductDatabase::loadModel(const std::string& filepath)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
        return false;

    records.clear();
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (uint32_t i = 0; i < count; ++i) {
        ProductRecord item;
        ifs.read(reinterpret_cast<char*>(&item.tempId), sizeof(item.tempId));

        ifs.read(reinterpret_cast<char*>(&item.id), sizeof(item.id));

        uint32_t nameLen = 0;
        ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        item.name.resize(nameLen);
        ifs.read(&item.name[0], nameLen);

        uint32_t codeLen = 0;
        ifs.read(reinterpret_cast<char*>(&codeLen), sizeof(codeLen));
        item.barcode.resize(codeLen);
        ifs.read(&item.barcode[0], codeLen);

        item.feature.resize(ProductFeatureEngine::featureDim);
        ifs.read(reinterpret_cast<char*>(item.feature.data()), ProductFeatureEngine::featureDim * sizeof(float));

        ifs.read(reinterpret_cast<char*>(&item.matchCount), sizeof(item.matchCount));

        maxTempId = std::max(maxTempId, item.tempId);

        records.push_back(item);
    }

    rebuildMatrix();
    return true;
}

std::vector<SearchResult> ProductDatabase::search(const std::vector<float>& queryFeature, int topK)
{
    if (featureMatrix.empty() || queryFeature.size() != static_cast<size_t>(ProductFeatureEngine::featureDim)) {
        return {};
    }

    // 将 query 转为 featureDim x 1 列向量
    cv::Mat queryMat(ProductFeatureEngine::featureDim, 1, CV_32F, const_cast<float*>(queryFeature.data()));

    // 矩阵乘法： (N x featureDim) * (featureDim x 1) = (N x 1) 得分矩阵
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
    lastResults.clear();
    for (size_t i = 0; i < std::min<size_t>(topK, scoreIndexMap.size()); ++i) {
        size_t idx = scoreIndexMap[i].second;
        lastResults.push_back({ records[idx].tempId, records[idx].id,
            records[idx].name,
            records[idx].barcode,
            scoreIndexMap[i].first, idx });
    }
    return lastResults;
}
