#include "productfeatureengine.h"
#include <cmath>

FeatureWeights ProductFeatureEngine::m_featureWeights = { 0.4f, 1.5f, 1.0f };

std::pair<cv::Mat, cv::Mat> ProductFeatureEngine::cropROI(const cv::Mat& bg, const cv::Mat& fg, double minArea, bool rotation)
{
    if (bg.empty() || fg.empty() || bg.size() != fg.size())
        return { cv::Mat(), cv::Mat() };

    // 1. 通道统一 (BGR)
    cv::Mat bg3ch = bg, fg3ch = fg;
    if (bg.channels() == 4)
        cv::cvtColor(bg, bg3ch, cv::COLOR_BGRA2BGR);
    else if (bg.channels() == 1)
        cv::cvtColor(bg, bg3ch, cv::COLOR_GRAY2BGR);

    if (fg.channels() == 4)
        cv::cvtColor(fg, fg3ch, cv::COLOR_BGRA2BGR);
    else if (fg.channels() == 1)
        cv::cvtColor(fg, fg3ch, cv::COLOR_GRAY2BGR);

    // 2. 适当降低模糊核（改为 5x5 / 7x7）
    cv::Mat bgBlur, fgBlur;
    cv::GaussianBlur(bg3ch, bgBlur, cv::Size(5, 5), 0);
    cv::GaussianBlur(fg3ch, fgBlur, cv::Size(5, 5), 0);

    // 3. 多通道最大差分
    cv::Mat diffColor, diff;
    cv::absdiff(bgBlur, fgBlur, diffColor);
    std::vector<cv::Mat> channels;
    cv::split(diffColor, channels);
    cv::max(channels[0], channels[1], diff);
    cv::max(diff, channels[2], diff);

    // 4. 二值化
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

    // 6. 形态学滤波
    cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kOpen);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kClose);

    // 7. 寻找最大轮廓【注意：必须用 CHAIN_APPROX_NONE 保证旋转一致性】
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

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

    const auto& bestContour = contours[maxIdx];

    // 8. 创建精确掩码 (不使用 convexHull 避免拉直笔套的凹槽特征)
    cv::Mat productMask = cv::Mat::zeros(fg3ch.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> hulls = { bestContour };
    cv::drawContours(productMask, hulls, 0, cv::Scalar(255), cv::FILLED);

    // 抠出前景
    cv::Mat foregroundOnly;
    fg3ch.copyTo(foregroundOnly, productMask);

    if (rotation) {
        // 9. 【核心改进】：计算最小外接矩形 (RotatedRect)，旋转摆正图像
        cv::RotatedRect minRect = cv::minAreaRect(bestContour);
        float angle = minRect.angle;
        cv::Size2f rectSize = minRect.size;

        // 确保长边始终在水平方向 (Swapping Width & Height if needed)
        if (rectSize.width < rectSize.height) {
            std::swap(rectSize.width, rectSize.height);
            angle += 90.0f;
        }

        // 构建仿射变换矩阵，绕中心旋转摆正
        cv::Mat M = cv::getRotationMatrix2D(minRect.center, angle, 1.0);

        // 调整旋转中心，使其居中平移到新的图像坐标系中
        M.at<double>(0, 2) += rectSize.width / 2.0 - minRect.center.x;
        M.at<double>(1, 2) += rectSize.height / 2.0 - minRect.center.y;

        cv::Mat croppedProduct, croppedMask;
        // 双线性插值旋转前景
        cv::warpAffine(foregroundOnly, croppedProduct, M, rectSize, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        // 最近邻插值旋转 Mask
        cv::warpAffine(productMask, croppedMask, M, rectSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // 返回经过【摆正化】处理的标准水平方向 ROI 和 Mask
        return { croppedProduct, croppedMask };
    } else {
        return { foregroundOnly, productMask };
    }
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

std::vector<float> ProductFeatureEngine::extractShapeFeature(const cv::Mat& roi, const cv::Mat& mask)
{
    // std::vector<float> huVec = extractHuShapeFeature(roi); // 确保内部做过 -log10(|hu|)
    std::vector<float> fourierVec = extractFourierShapeFeature(roi, mask);
    std::vector<float> shapeVec;
    shapeVec.reserve(fourierVec.size());
    shapeVec.insert(shapeVec.end(), fourierVec.begin(), fourierVec.end());
    // shapeVec.insert(shapeVec.end(), huVec.begin(), huVec.end());

    // 全局 L2 归一化
    normalizeVector(shapeVec);

    return shapeVec;
}

std::vector<float> ProductFeatureEngine::extractHuShapeFeature(const cv::Mat& roi)
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

// 按弧长均匀重采样轮廓到 N 个点
// 目的：让同一形状无论缩放/原始点数多少，都得到同样的采样，|F(k)| 分布稳定
inline std::vector<cv::Point2f> resampleContourUniform(
    const std::vector<cv::Point>& contour, int N)
{
    std::vector<cv::Point2f> pts;
    if (contour.size() < 2 || N <= 0)
        return pts;

    // 累积弧长
    std::vector<float> cumlen(contour.size() + 1, 0.0f);
    for (size_t i = 0; i < contour.size(); ++i) {
        size_t j = (i + 1) % contour.size();
        const cv::Point2f a = contour[i];
        const cv::Point2f b = contour[j];
        cumlen[i + 1] = cumlen[i] + std::hypot(b.x - a.x, b.y - a.y);
    }
    const float perimeter = cumlen.back();
    if (perimeter < 1e-6f)
        return pts;

    pts.reserve(N);
    const float step = perimeter / static_cast<float>(N);
    int seg = 0;
    for (int k = 0; k < N; ++k) {
        const float target = k * step;
        while (seg + 1 < static_cast<int>(cumlen.size()) && cumlen[seg + 1] < target)
            ++seg;
        const float seg_len = cumlen[seg + 1] - cumlen[seg];
        const float t = (seg_len > 1e-6f) ? (target - cumlen[seg]) / seg_len : 0.0f;
        const cv::Point2f a = contour[seg % contour.size()];
        const cv::Point2f b = contour[(seg + 1) % contour.size()];
        pts.push_back(a + t * (b - a));
    }
    return pts;
}

std::vector<float> ProductFeatureEngine::extractFourierShapeFeature(const cv::Mat& roi, const cv::Mat& mask, int numCoeffs)
{
    std::vector<float> feature(std::max(numCoeffs, 1), 0.0f);
    if (roi.empty() || numCoeffs <= 0)
        return feature;

    cv::Mat binary;
    if (!mask.empty() && mask.size() == roi.size()) {
        cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);
    } else {
        cv::Mat gray;
        if (roi.channels() == 3)
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else
            gray = roi;
        cv::threshold(gray, binary, 0, 255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    // 形态学去噪（开运算去小噪点，闭运算补小洞）
    const cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 });
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, k3);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, k3);

    // ---- 2. 找轮廓，取面积最大的 ----
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE); // SIMPLE 比 NONE 更稳

    std::vector<cv::Point> mainContour;
    double max_area = 0.0;
    for (auto& c : contours) {
        const double a = cv::contourArea(c);
        if (a > max_area) {
            max_area = a;
            mainContour = c;
        }
    }
    const double min_area = 0.01 * binary.total();
    if (mainContour.size() < 4 || max_area < min_area)
        return feature;

    // ---- 3. 按弧长重采样到固定点数 ----
    const int FIXED_N = 128;
    std::vector<cv::Point2f> resampled = resampleContourUniform(mainContour, FIXED_N);
    if (static_cast<int>(resampled.size()) < 4)
        return feature;

    // ---- 4. 质心平移（平移不变性）----
    cv::Point2f centroid(0.f, 0.f);
    for (auto& p : resampled)
        centroid += p;
    centroid *= 1.0f / static_cast<float>(resampled.size());

    cv::Mat complexContour(FIXED_N, 1, CV_32FC2);
    for (int i = 0; i < FIXED_N; ++i) {
        complexContour.at<cv::Vec2f>(i, 0) = cv::Vec2f(resampled[i].x - centroid.x,
            resampled[i].y - centroid.y);
    }

    // ---- 5. DFT ----
    cv::Mat dftResult;
    cv::dft(complexContour, dftResult, cv::DFT_COMPLEX_OUTPUT);

    // ---- 6. 取正频率 1..N/2，|F(k)| 做旋转不变 ----
    //       尺度不变由最后 L2 归一化统一处理，不再用 |F(1)| 做 scaleFactor
    const int maxCoeffs = std::min(numCoeffs, FIXED_N / 2);

    float sum_mag = 0.0f;
    for (int i = 0; i < maxCoeffs; ++i) {
        const cv::Vec2f c = dftResult.at<cv::Vec2f>(i + 1, 0);
        feature[i] = std::sqrt(c[0] * c[0] + c[1] * c[1]);
        sum_mag += feature[i];
    }

    // ---- 7. 去均值（关键：解决“全正分量 → 都像”）----
    //       只对有效位去均值，后面的补 0 位保持 0
    if (maxCoeffs > 0) {
        const float mean = sum_mag / static_cast<float>(maxCoeffs);
        for (int i = 0; i < maxCoeffs; ++i)
            feature[i] -= mean;
    }

    // ---- 8. L2 归一化（模长为 1）----
    normalizeVector(feature);

    return feature;
}

std::vector<float> ProductFeatureEngine::extractTextureFeature(const cv::Mat& roi, const cv::Mat& croppedMask)
{
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    // ---- 1. mask 二值化 + erode 去边缘 ----
    cv::Mat mask_bin;
    cv::threshold(croppedMask, mask_bin, 127, 255, cv::THRESH_BINARY);
    cv::Mat mask_eroded;
    cv::erode(mask_bin, mask_eroded, cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 }));

    // ---- 2. 计算 LBP（8 邻域）----
    cv::Mat lbp(gray.rows - 2, gray.cols - 2, CV_8UC1, cv::Scalar(0));
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

    // ---- 3. Uniform LBP 映射（256 -> 59）----
    static const std::array<int, 256> uniform_map = [] {
        std::array<int, 256> m {};
        int idx = 0;
        for (int code = 0; code < 256; ++code) {
            int trans = 0;
            for (int i = 0; i < 8; ++i) {
                int cur = (code >> i) & 1;
                int nxt = (code >> ((i + 1) % 8)) & 1;
                if (cur != nxt)
                    ++trans;
            }
            m[code] = (trans <= 2) ? idx++ : 58;
        }
        return m;
    }();

    // 把 lbp 映射到 uniform 空间
    cv::Mat lbp_u(lbp.size(), CV_8UC1);
    for (int r = 0; r < lbp.rows; ++r) {
        const uchar* src = lbp.ptr<uchar>(r);
        uchar* dst = lbp_u.ptr<uchar>(r);
        for (int c = 0; c < lbp.cols; ++c)
            dst[c] = static_cast<uchar>(uniform_map[src[c]]);
    }

    // ---- 4. 直方图（59 维），用 erode 后的 mask ----
    //    lbp 是 (rows-2, cols-2)，mask_eroded 是 (rows, cols)，crop 对齐
    cv::Rect lbp_rect(1, 1, lbp.cols, lbp.rows);
    cv::Mat mask_for_hist = mask_eroded(lbp_rect);

    int histSize = 59;
    float range[] = { 0, 59 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&lbp_u, 1, 0, mask_for_hist, hist, 1, &histSize, &histRange);

    // ---- 5. 检查有效性 ----
    double total = cv::sum(hist)[0];
    if (total < 1e-6)
        return std::vector<float>(59, 0.0f);

    // ---- 6. 去均值（解决全正问题）----
    std::vector<float> vec(59);
    for (int i = 0; i < 59; ++i)
        vec[i] = hist.at<float>(i);
    float mean = 0;
    for (float v : vec)
        mean += v;
    mean /= 59.f;
    for (float& v : vec)
        v -= mean;

    // ---- 7. L2 归一化 ----
    normalizeVector(vec);

    return vec;
}

std::vector<float> ProductFeatureEngine::extractVisual(const cv::Mat& roi, const cv::Mat& croppedMask)
{
    if (roi.empty())
        return {};

    auto colorVec = extractColorFeature(roi, croppedMask); // 128
    auto shapeVec = extractShapeFeature(roi, croppedMask); // 64
    auto textureVec = extractTextureFeature(roi, croppedMask); // 256

    std::vector<float> feature;
    feature.reserve(colorVec.size() + shapeVec.size() + textureVec.size());

    // 加权融合
    for (float v : colorVec)
        feature.push_back(v * m_featureWeights.color);
    for (float v : shapeVec)
        feature.push_back(v * m_featureWeights.shape);
    for (float v : textureVec)
        feature.push_back(v * m_featureWeights.texture);

    // 全局 L2 归一化
    normalizeVector(feature);

    return feature;
}

void ProductFeatureEngine::reinforceFeatureVector(std::vector<float>& dbVec, const std::vector<float>& queryVec, float alpha, float beta)
{
    if (dbVec.size() != queryVec.size())
        return;

    for (size_t i = 0; i < dbVec.size(); ++i) {
        // 融合特征
        dbVec[i] = alpha * dbVec[i] + beta * queryVec[i];
    }

    normalizeVector(dbVec);
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

std::vector<float> ProductFeatureEngine::fuseFeatures(const cv::Mat& roi, const cv::Mat& croppedMask, const std::string& text, float wVisual, float wText)
{
    std::vector<float> fused;
    fused.reserve(featureDim);

    auto visualFeature = extractVisual(roi, croppedMask);

    // A. 追加并加权图片特征 (VISUAL_DIM 维)
    for (size_t i = 0; i < VISUAL_DIM && i < visualFeature.size(); ++i) {
        fused.push_back(visualFeature[i] * wVisual);
    }

    // B. 提取并加权名称特征 (TEXT_DIM 维)
    std::vector<float> textVec = extractTextFeature(text);
    for (float val : textVec) {
        fused.push_back(val * wText);
    }

    // C. 整体重新进行 L2 归一化，使得矩阵乘法点积结果仍处于 [0.0, 1.0] 范围内
    normalizeVector(fused);
    return fused;
}

void ProductFeatureEngine::normalizeVector(std::vector<float>& vec)
{
    float sumSq = 0.0f;
    for (float v : vec)
        sumSq += v * v;
    float norm = std::sqrt(sumSq);
    if (norm > 1e-6f) {
        for (float& v : vec)
            v /= norm;
    }
}

std::vector<float> ProductFeatureEngine::extractTextFeature(const std::string& name)
{
    std::vector<float> textVec(TEXT_DIM, 0.0f);
    if (name.empty())
        return textVec;

    std::vector<std::string> charList;
    charList.reserve(name.size());

    size_t i = 0;
    while (i < name.size()) {
        unsigned char c = name[i];
        size_t charLen = 1;

        // 根据 UTF-8 首字节判断当前字符占用的字节长度
        if ((c & 0x80) == 0)
            charLen = 1; // ASCII
        else if ((c & 0xE0) == 0xC0)
            charLen = 2; // 2字节字符
        else if ((c & 0xF0) == 0xE0)
            charLen = 3; // 常见汉字（3字节）
        else if ((c & 0xF8) == 0xF0)
            charLen = 4; // 4字节字符（如Emoji）

        // 边界防护
        if (i + charLen > name.size())
            break;

        charList.push_back(name.substr(i, charLen));
        i += charLen;
    }

    // 提取单字与双字 N-Gram 并映射到 Hash 桶中
    std::hash<std::string> hasher;

    // 第二步：基于 UTF-8 字符列表提取 Unigram 和 Bigram
    for (size_t i = 0; i < charList.size(); ++i) {
        // --- 1. Unigram (单字) ---
        const std::string& unigram = charList[i];
        size_t bucket1 = hasher(unigram) % TEXT_DIM;
        textVec[bucket1] += 1.0f;

        // --- 2. Bigram (双字词) ---
        if (i + 1 < charList.size()) {
            std::string bigram = charList[i] + charList[i + 1];
            size_t bucket2 = hasher(bigram) % TEXT_DIM;
            textVec[bucket2] += 1.5f; // 连续双字赋予更高权重
        }
    }

    normalizeVector(textVec); // 归一化

    return textVec;
}
