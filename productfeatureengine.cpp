#include "productfeatureengine.h"
#include <cmath>
#include <qDebug>

std::pair<cv::Mat, cv::Mat> ProductFeatureEngine::cropROI(const cv::Mat& bg, const cv::Mat& fg, double minArea)
{
    if (bg.empty() || fg.empty() || bg.size() != fg.size())
        return { cv::Mat(), cv::Mat() };

    // 1. 通道统一 (BGR)
    cv::Mat bg3ch, fg3ch;
    if (bg.channels() == 4)
        cv::cvtColor(bg, bg3ch, cv::COLOR_BGRA2BGR);
    else if (bg.channels() == 1)
        cv::cvtColor(bg, bg3ch, cv::COLOR_GRAY2BGR);
    else
        bg3ch = bg;

    if (fg.channels() == 4)
        cv::cvtColor(fg, fg3ch, cv::COLOR_BGRA2BGR);
    else if (fg.channels() == 1)
        cv::cvtColor(fg, fg3ch, cv::COLOR_GRAY2BGR);
    else
        fg3ch = fg;

    // 2. 高斯模糊降噪 (15x15)
    cv::Mat bgBlur, fgBlur;
    cv::GaussianBlur(bg3ch, bgBlur, cv::Size(15, 15), 0);
    cv::GaussianBlur(fg3ch, fgBlur, cv::Size(15, 15), 0);

    // 3. 多通道最大差分
    cv::Mat diffColor, diff;
    cv::absdiff(bgBlur, fgBlur, diffColor);
    std::vector<cv::Mat> channels;
    cv::split(diffColor, channels);
    cv::max(channels[0], channels[1], diff);
    cv::max(diff, channels[2], diff);

    // 4. 二值化 (阈值 40~45)
    cv::Mat thresh;
    cv::threshold(diff, thresh, 42, 255, cv::THRESH_BINARY);

    // 5. 边缘切除 (15px)
    int border = 15;
    if (thresh.rows > border * 2 && thresh.cols > border * 2) {
        thresh(cv::Rect(0, 0, thresh.cols, border)).setTo(0);
        thresh(cv::Rect(0, thresh.rows - border, thresh.cols, border)).setTo(0);
        thresh(cv::Rect(0, 0, border, thresh.rows)).setTo(0);
        thresh(cv::Rect(thresh.cols - border, 0, border, thresh.rows)).setTo(0);
    }

    // 6. 减小形态学核，防止商品与底部的线缆粘连
    cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat kClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 降至 5x5
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kOpen);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kClose);

    // 7. 寻找最大轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int maxIdx = -1;
    double maxArea = 0.0;
    double imgArea = static_cast<double>(fg3ch.cols * fg3ch.rows);

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > minArea && area < imgArea * 0.80 && area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx == -1)
        return { cv::Mat(), cv::Mat() };

    // 1. 提取目标轮廓与外接矩形
    const auto& bestContour = contours[maxIdx];
    cv::Rect bestRect = cv::boundingRect(bestContour);
    bestRect &= cv::Rect(0, 0, fg3ch.cols, fg3ch.rows);

    // 2. 创建精细掩码（Mask）：只将商品轮廓区域设为白色，且填充内部所有空洞
    cv::Mat productMask = cv::Mat::zeros(fg3ch.size(), CV_8UC1);

    // 使用凸包（Convex Hull）把不规则果皮的内部空洞和凹陷补齐
    std::vector<cv::Point> hull;
    cv::convexHull(bestContour, hull);
    std::vector<std::vector<cv::Point>> hulls = { hull };
    cv::drawContours(productMask, hulls, 0, cv::Scalar(255), cv::FILLED);

    // 3. 【核心优化】背景遮罩抠图：只保留商品本身，背景木纹与黑线直接置黑 (0,0,0)
    cv::Mat foregroundOnly;
    fg3ch.copyTo(foregroundOnly, productMask);

    // 4. 裁切最终 ROI（此时框外的黑线/木纹被 Mask 遮罩过滤掉了，极利于后续 HSV/LBP 提取）
    cv::Mat croppedProduct = foregroundOnly(bestRect).clone();

    return { croppedProduct, productMask(bestRect).clone() };
}

std::vector<float> ProductFeatureEngine::extractColorFeature(const cv::Mat& roi, const cv::Mat& croppedMask)
{
    std::vector<float> vec;
    vec.reserve(128);

    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int hBins = 8, sBins = 4;
    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 }, sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };
    int channels[] = { 0, 1 };

    int cellW = roi.cols / 2, cellH = roi.rows / 2;

    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            cv::Rect rect(c * cellW, r * cellH,
                (c == 1) ? (roi.cols - cellW) : cellW,
                (r == 1) ? (roi.rows - cellH) : cellH);

            cv::Mat cellHSV = hsv(rect);
            cv::Mat cellMask = croppedMask(rect);
            cv::Mat hist;

            if (cv::countNonZero(cellMask) > 0) {
                cv::calcHist(&cellHSV, 1, channels, cellMask, hist, 2, histSize, ranges);
                cv::normalize(hist, hist, 1, 0, cv::NORM_L2);
            } else {
                hist = cv::Mat::zeros(histSize[0], histSize[1], CV_32F);
            }

            for (int i = 0; i < hist.rows; ++i) {
                for (int j = 0; j < hist.cols; ++j) {
                    vec.push_back(hist.at<float>(i, j));
                }
            }
        }
    }
    return vec;
}

std::vector<float> ProductFeatureEngine::extractShapeFeature(const cv::Mat& roi)
{
    std::vector<float> vec;
    vec.reserve(7);

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    cv::Moments m = cv::moments(gray, false);
    double hu[7];
    cv::HuMoments(m, hu);

    for (int i = 0; i < 7; ++i) {
        double sign = (hu[i] < 0) ? -1.0 : 1.0;
        double val = (std::abs(hu[i]) > 1e-10) ? (sign * std::log10(std::abs(hu[i]))) : 0.0;
        vec.push_back(static_cast<float>(val));
    }
    return vec;
}

std::vector<float> ProductFeatureEngine::extractTextureFeature(const cv::Mat& roi, const cv::Mat& croppedMask)
{
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);
    for (int r = 1; r < gray.rows - 1; ++r) {
        const uchar* pPrev = gray.ptr<uchar>(r - 1);
        const uchar* pCurr = gray.ptr<uchar>(r);
        const uchar* pNext = gray.ptr<uchar>(r + 1);
        uchar* pLbp = lbp.ptr<uchar>(r - 1);

        for (int c = 1; c < gray.cols - 1; ++c) {
            uchar center = pCurr[c];
            uchar code = 0;
            code |= (pPrev[c - 1] >= center) << 7;
            code |= (pPrev[c] >= center) << 6;
            code |= (pPrev[c + 1] >= center) << 5;
            code |= (pCurr[c + 1] >= center) << 4;
            code |= (pNext[c + 1] >= center) << 3;
            code |= (pNext[c] >= center) << 2;
            code |= (pNext[c - 1] >= center) << 1;
            code |= (pCurr[c - 1] >= center) << 0;
            pLbp[c - 1] = code;
        }
    }

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&lbp, 1, 0, croppedMask(cv::Rect(1, 1, lbp.cols, lbp.rows)), hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L2);

    std::vector<float> vec(256);
    for (int i = 0; i < 256; ++i)
        vec[i] = hist.at<float>(i);
    return vec;
}

std::vector<float> ProductFeatureEngine::extract(const cv::Mat& roi, const cv::Mat& croppedMask, FeatureWeights weights)
{
    if (roi.empty())
        return {};

    auto colorVec = extractColorFeature(roi, croppedMask); // 128
    auto shapeVec = extractShapeFeature(roi); // 7
    auto textureVec = extractTextureFeature(roi, croppedMask); // 256

    std::vector<float> feature;
    feature.reserve(colorVec.size() + shapeVec.size() + textureVec.size());

    // 加权融合
    for (float v : colorVec)
        feature.push_back(v * weights.color);
    for (float v : shapeVec)
        feature.push_back(v * weights.shape);
    for (float v : textureVec)
        feature.push_back(v * weights.texture);

    // 全局 L2 归一化
    float sumSq = 0.0f;
    for (float v : feature)
        sumSq += v * v;
    float norm = std::sqrt(sumSq);

    if (norm > 1e-6f) {
        for (float& v : feature)
            v /= norm;
    }

    return feature;
}

float ProductFeatureEngine::computeSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB)
{
    if (vecA.size() != vecB.size() || vecA.empty())
        return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < vecA.size(); ++i) {
        dot += vecA[i] * vecB[i];
    }
    return dot;
}
