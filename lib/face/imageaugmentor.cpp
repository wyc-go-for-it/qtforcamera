#include "ImageAugmentor.h"
#include <cmath>

// 1. 亮度调整
cv::Mat ImageAugmentor::adjustBrightness(const cv::Mat& src, int delta)
{
    cv::Mat dst;
    src.convertTo(dst, -1, 1.0, delta); // dst = src + delta
    return dst;
}

// 2. 对比度调整
cv::Mat ImageAugmentor::adjustContrast(const cv::Mat& src, float alpha)
{
    cv::Mat dst;
    src.convertTo(dst, -1, alpha, 0); // dst = src * alpha
    return dst;
}

// 3. 亮度+对比度
cv::Mat ImageAugmentor::adjustBrightContrast(const cv::Mat& src, float alpha, int beta)
{
    cv::Mat dst;
    src.convertTo(dst, -1, alpha, beta); // dst = src * alpha + beta
    return dst;
}

// 4. Gamma 校正
cv::Mat ImageAugmentor::adjustGamma(const cv::Mat& src, double gamma)
{
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat dst;
    cv::LUT(src, lut, dst);
    return dst;
}

// 5. HSV 调整
cv::Mat ImageAugmentor::adjustHSV(const cv::Mat& src, int hShift, float sScale, float vScale)
{
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch(3);
    cv::split(hsv, ch);

    // 色相偏移 (OpenCV 中 H 范围为 0~179)
    if (hShift != 0) {
        int shift = hShift % 180;
        if (shift < 0)
            shift += 180;
        for (int y = 0; y < ch[0].rows; ++y) {
            uchar* row = ch[0].ptr<uchar>(y);
            for (int x = 0; x < ch[0].cols; ++x) {
                row[x] = (row[x] + shift) % 180;
            }
        }
    }
    // 饱和度和明度缩放
    if (sScale != 1.0f)
        ch[1].convertTo(ch[1], -1, sScale, 0);
    if (vScale != 1.0f)
        ch[2].convertTo(ch[2], -1, vScale, 0);

    cv::merge(ch, hsv);
    cv::Mat dst;
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
    return dst;
}

// 6. 高斯噪声
cv::Mat ImageAugmentor::addGaussianNoise(const cv::Mat& src, double mean, double stddev)
{
    cv::Mat noise(src.size(), src.type());
    cv::randn(noise, mean, stddev);
    cv::Mat dst;
    cv::add(src, noise, dst);
    return dst;
}

// 7. 高斯模糊
cv::Mat ImageAugmentor::addBlur(const cv::Mat& src, int ksize)
{
    if (ksize < 1)
        ksize = 1;
    if (ksize % 2 == 0)
        ksize++; // 确保是奇数
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0);
    return dst;
}

// 8. 随机组合增强
cv::Mat ImageAugmentor::randomPixelAugment(const cv::Mat& src, cv::RNG& rng)
{
    cv::Mat img = src.clone();

    // 亮度
    if (rng.uniform(0, 100) < 50) {
        img = adjustBrightness(img, rng.uniform(-40, 40));
    }
    // 对比度
    if (rng.uniform(0, 100) < 30) {
        img = adjustContrast(img, rng.uniform(0.7f, 1.3f));
    }
    // Gamma
    if (rng.uniform(0, 100) < 30) {
        img = adjustGamma(img, rng.uniform(0.7, 1.4));
    }
    // HSV
    if (rng.uniform(0, 100) < 20) {
        img = adjustHSV(img, rng.uniform(-10, 10), rng.uniform(0.8f, 1.2f), rng.uniform(0.85f, 1.15f));
    }
    // 噪声
    if (rng.uniform(0, 100) < 20) {
        img = addGaussianNoise(img, 0, rng.uniform(5.0, 15.0));
    }
    // 模糊
    if (rng.uniform(0, 100) < 20) {
        img = addBlur(img, 3);
    }

    return img;
}

// 9. 水平翻转 (含 BBox 同步修改)
cv::Mat ImageAugmentor::flipHorizontal(cv::Mat& src, std::vector<cv::Rect>& bboxes)
{
    cv::Mat dst;
    cv::flip(src, dst, 1); // 1 表示水平翻转
    int w = src.cols;

    for (auto& b : bboxes) {
        b.x = w - b.x - b.width;
    }
    return dst;
}

// 10. 随机旋转 (测试人脸对齐/姿态鲁棒性)
cv::Mat ImageAugmentor::rotate(const cv::Mat& src, float angle)
{
    int h = src.rows;
    int w = src.cols;
    cv::Point2f center(w / 2.0f, h / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst;
}

// 11. 随机缩放 (模拟远近)
cv::Mat ImageAugmentor::scale(const cv::Mat& src, float factor)
{
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), factor, factor);
    return dst;
}
