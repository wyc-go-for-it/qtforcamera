

#include "BboxU2netp.h"
#include "opencv2/imgcodecs.hpp"
#include <cstring>
#include <opencv2/imgproc.hpp>

#ifdef ENABLE_LOG
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "HandDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...)                 \
    do {                          \
        std::printf(__VA_ARGS__); \
        fflush(stdout);           \
    } while (0)
#define LOGE(...)                          \
    do {                                   \
        std::fprintf(stderr, __VA_ARGS__); \
        fflush(stderr);                    \
    } while (0)
#endif
#else
#define LOGI(...) ((void)0)
#define LOGE(...) ((void)0)
#endif

// ---------------- 构造 / 析构 ----------------

BboxU2netp::BboxU2netp() = default;

BboxU2netp::BboxU2netp(const Config& cfg)
    : cfg_(cfg)
{
}

std::pair<cv::Mat, cv::Mat> BboxU2netp::u2netpCropROI(const cv::Mat& fg, double minArea, bool rotation)
{

    if (fg.empty() || !isLoaded())
        return { fg, cv::Mat() };

    cv::Mat fg3ch = fg;
    if (fg.channels() == 4)
        cv::cvtColor(fg, fg3ch, cv::COLOR_BGRA2BGR);
    else if (fg.channels() == 1)
        cv::cvtColor(fg, fg3ch, cv::COLOR_GRAY2BGR);

    cv::Mat mask = segment(fg3ch);
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

    static int index = 22;

    auto name = "./wyc_img_data/" + std::to_string(index++);
    cv::imwrite(name + ".png", fg3ch);
    cv::imwrite(name + ".jpg", productMask);

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

BboxU2netp::~BboxU2netp() = default;

// ---------------- 加载 ----------------

static void setupNet(ncnn::Net& net, const BboxU2netp::Config& cfg)
{
    net.opt.use_vulkan_compute = cfg.useVulkan;
    net.opt.num_threads = cfg.numThreads;
    net.opt.use_fp16_packed = cfg.useFP16;
    net.opt.use_fp16_storage = cfg.useFP16;

#if defined(__ANDROID__)
    net.opt.use_fp16_arithmetic = true;
#else
    net.opt.use_fp16_arithmetic = false;
#endif

    net.opt.blob_allocator = 0;
    net.opt.workspace_allocator = 0;
    net.opt.openmp_blocktime = 0;

    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;

    net.opt.lightmode = true;
}

bool BboxU2netp::load_model(const std::string& paramPath,
    const std::string& binPath)
{
    std::lock_guard<std::mutex> lk(mtx_);

    setupNet(net_, cfg_);

    if (net_.load_param(paramPath.c_str()) != 0) {
        LOGE("BboxU2netp load_param failed: %s", paramPath.c_str());
        return false;
    }
    if (net_.load_model(binPath.c_str()) != 0) {
        LOGE("BboxU2netp load_model failed: %s", binPath.c_str());
        return false;
    }
    loaded_ = true;
    LOGI("BboxU2netp model loaded from file");
    return true;
}

bool BboxU2netp::isLoaded() const { return loaded_; }

cv::Mat BboxU2netp::runNetwork(const cv::Mat& bgr)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
        bgr.cols, bgr.rows, cfg_.inputSize, cfg_.inputSize);

    const float mean[3] = { 123.675f, 116.28f, 103.53f };
    const float norm[3] = { 1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f };

    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net_.create_extractor();
    ex.input(cfg_.inputName.c_str(), in);

    ncnn::Mat out;
    if (ex.extract(cfg_.outputName.c_str(), out) != 0) {
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

    LOGI("runNetwork in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,推理:%.2f\n", in.w, in.h, in.c, out.h, out.w, std::chrono::duration<double, std::milli>(t1 - t0).count());

    double mn, mx;
    cv::minMaxLoc(raw, &mn, &mx);
    LOGI("mask min=%f max=%f mean=%f", mn, mx, cv::mean(raw)[0]);

    fflush(stdout);

    return raw;
}

cv::Mat BboxU2netp::postprocess(const cv::Mat& raw, const cv::Size& target)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat m = raw;

    if (cfg_.applySigmoid) {
        cv::Mat neg;
        cv::exp(-m, neg);
        m = 1.0 / (1.0 + neg);
    }

    cv::Mat resized;
    cv::resize(m, resized, target, 0, 0, cv::INTER_LINEAR);

    if (cfg_.binaryMask) {
        cv::threshold(resized, resized, cfg_.threshold, 1.0, cv::THRESH_BINARY);
    }

    /* 模糊会影响特征提取
    if (m_cfg.featherSigma > 0.f) {
        cv::GaussianBlur(resized, resized, cv::Size(0, 0), m_cfg.featherSigma);
    } */

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI("postprocess 推理:%.2f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    return resized;
}

cv::Mat BboxU2netp::segment(const cv::Mat& bgr)
{
    if (!loaded_) {
        LOGE("model not loaded");
        return {};
    }
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        LOGE("invalid input: must be CV_8UC3");
        return {};
    }

    std::lock_guard<std::mutex> lk(mtx_);
    cv::Mat raw = runNetwork(bgr);
    if (raw.empty())
        return {};

    return postprocess(raw, bgr.size());
}

cv::Mat BboxU2netp::compose(const cv::Mat& bgr,
    const cv::Mat& mask,
    const cv::Scalar& bg)
{
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_32FC1);
    CV_Assert(bgr.size() == mask.size());

    cv::Mat dst(bgr.size(), CV_8UC3);
    for (int y = 0; y < bgr.rows; ++y) {
        const uchar* fg = bgr.ptr<uchar>(y);
        const float* a = mask.ptr<float>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const float av = a[x];
            const float iv = 1.f - av;
            d[x * 3 + 0] = cv::saturate_cast<uchar>(fg[x * 3 + 0] * av + bg[0] * iv);
            d[x * 3 + 1] = cv::saturate_cast<uchar>(fg[x * 3 + 1] * av + bg[1] * iv);
            d[x * 3 + 2] = cv::saturate_cast<uchar>(fg[x * 3 + 2] * av + bg[2] * iv);
        }
    }
    return dst;
}

cv::Mat BboxU2netp::process(const cv::Mat& bgr, const cv::Scalar& bg)
{
    cv::Mat mask = segment(bgr);
    if (mask.empty())
        return {};
    return compose(bgr, mask, bg);
}

cv::Mat BboxU2netp::processWithAlpha(const cv::Mat& bgr)
{
    cv::Mat mask = segment(bgr);
    if (mask.empty())
        return {};

    cv::Mat bgra(bgr.size(), CV_8UC4);
    for (int y = 0; y < bgr.rows; ++y) {
        const uchar* fg = bgr.ptr<uchar>(y);
        const float* a = mask.ptr<float>(y);
        uchar* d = bgra.ptr<uchar>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            d[x * 4 + 0] = fg[x * 3 + 0];
            d[x * 4 + 1] = fg[x * 3 + 1];
            d[x * 4 + 2] = fg[x * 3 + 2];
            d[x * 4 + 3] = cv::saturate_cast<uchar>(a[x] * 255.f);
        }
    }
    return bgra;
}
