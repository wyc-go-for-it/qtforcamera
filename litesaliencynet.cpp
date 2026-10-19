#include "litesaliencynet.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "U2NetP"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...)                 \
    do {                          \
        std::printf(__VA_ARGS__); \
        std::printf("\n");        \
        fflush(stdout);           \
    } while (0)
#define LOGE(...)                          \
    do {                                   \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");        \
        fflush(stdout);                    \
    } while (0)
#endif

LiteSaliencyNet::LiteSaliencyNet()
{
}

bool LiteSaliencyNet::load_model(const std::string& paramPath, const std::string& binPath)
{
    m_net.opt.use_vulkan_compute = m_cfg.useVulkan;
    m_net.opt.num_threads = m_cfg.numThreads;
    m_net.opt.use_fp16_packed = m_cfg.useFP16;
    m_net.opt.use_fp16_storage = m_cfg.useFP16;

#if defined(__ANDROID__)
    m_net.opt.use_fp16_arithmetic = true;
#else
    m_net.opt.use_fp16_arithmetic = false;
#endif

    m_net.opt.blob_allocator = 0;
    m_net.opt.workspace_allocator = 0;
    m_net.opt.openmp_blocktime = 0;

    m_net.opt.use_packing_layout = true;
    m_net.opt.use_winograd_convolution = true;
    m_net.opt.use_sgemm_convolution = true;

    m_net.opt.lightmode = true;

    if (m_net.load_param(paramPath.c_str()) != 0) {
        LOGE("LiteSaliencyNet load_param failed: %s", paramPath.c_str());
        return false;
    }
    if (m_net.load_model(binPath.c_str()) != 0) {
        LOGE("LiteSaliencyNet load_model failed: %s", binPath.c_str());
        return false;
    }
    m_loaded = true;
    LOGI("LiteSaliencyNet model loaded from file");
    return true;
}

std::pair<cv::Mat, cv::Mat> LiteSaliencyNet::liteSaliencyCropROI(const cv::Mat& fg, double minArea, bool rotation)
{
    if (fg.empty() || !m_loaded)
        return { cv::Mat(), cv::Mat() };

    cv::Mat fg3ch = fg;
    if (fg.channels() == 4)
        cv::cvtColor(fg, fg3ch, cv::COLOR_BGRA2BGR);
    else if (fg.channels() == 1)
        cv::cvtColor(fg, fg3ch, cv::COLOR_GRAY2BGR);

    cv::Mat mask = createMask(fg3ch);
    if (mask.empty()) {
        return { cv::Mat(), mask };
    }

    cv::Mat productMask = mask;
    if (!config().binaryMask) {
        cv::threshold(mask, productMask, 0.5, 255, cv::THRESH_BINARY);
    }
    productMask.convertTo(mask, CV_8UC1, 255.0);

    productMask = mask;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(productMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
        return { cv::Mat(), cv::Mat() };

    std::vector<cv::Point> allPts;
    for (const auto& c : contours) {
        if (cv::contourArea(c) > minArea) {
            allPts.insert(allPts.end(), c.begin(), c.end());
        }
    }

    cv::rectangle(fg,
        cv::boundingRect(allPts) & cv::Rect(0, 0, fg.cols, fg.rows),
        cv::Scalar(0, 0, 255),
        2,
        cv::LINE_8);

    cv::Mat foregroundOnly;
    fg3ch.copyTo(foregroundOnly, productMask);

    if (rotation) {
        // 【核心改进】：计算最小外接矩形 (RotatedRect)，旋转摆正图像
        cv::RotatedRect minRect = cv::minAreaRect(allPts);
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

cv::Mat LiteSaliencyNet::runNetwork(const cv::Mat& bgr)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
        bgr.cols, bgr.rows, m_cfg.inputSize, m_cfg.inputSize);

    const float mean[3] = { 123.675f, 116.28f, 103.53f };
    const float norm[3] = { 1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f };

    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = m_net.create_extractor();
    ex.input(m_cfg.inputName.c_str(), in);

    ncnn::Mat out;
    if (ex.extract(m_cfg.outputName.c_str(), out) != 0) {
        LOGE("extract failed");
        return {};
    }

    cv::Mat raw(out.h, out.w, CV_32FC1);
    if (out.elempack == 1) {
        std::memcpy(raw.data, out.data, out.total() * sizeof(float));
    } else {
        ncnn::Mat unpacked;
        ncnn::convert_packing(out, unpacked, 1);
        std::memcpy(raw.data, unpacked.data, unpacked.total() * sizeof(float));
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    LOGI("runNetwork bgr:%d x %d in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,推理:%.2f\n", bgr.cols, bgr.rows, in.w, in.h, in.c, out.h, out.w, std::chrono::duration<double, std::milli>(t1 - t0).count());

    double mn, mx;
    cv::minMaxLoc(raw, &mn, &mx);
    LOGI("mask min=%f max=%f mean=%f", mn, mx, cv::mean(raw)[0]);

    return raw;
}

cv::Mat LiteSaliencyNet::createMask(const cv::Mat& bgr)
{
    if (!m_loaded) {
        LOGE("model not loaded");
        return {};
    }
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        LOGE("invalid input: must be CV_8UC3");
        return {};
    }

    cv::Mat raw = runNetwork(bgr);
    if (raw.empty())
        return {};

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat m = raw;

    if (m_cfg.applySigmoid) {
        cv::Mat neg;
        cv::exp(-m, neg);
        m = 1.0 / (1.0 + neg);
    }

    const cv::Size& target = bgr.size();

    cv::Mat resized;
    cv::resize(m, resized, target, 0, 0, cv::INTER_LINEAR);

    if (m_cfg.binaryMask) {
        cv::threshold(resized, resized, m_cfg.threshold, 1.0, cv::THRESH_BINARY);
    }

    /* 模糊会影响特征提取
    if (m_cfg.featherSigma > 0.f) {
        cv::GaussianBlur(resized, resized, cv::Size(0, 0), m_cfg.featherSigma);
    } */

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI("LiteSaliencyNet resized :%.2f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    return resized;
}
