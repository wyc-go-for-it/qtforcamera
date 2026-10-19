#ifndef BBOXU2NETP_H
#define BBOXU2NETP_H

#include "opencv2/imgproc.hpp"
#pragma once

#include <ncnn/net.h>
#include <opencv2/core.hpp>

#include <mutex>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager.h>
#endif

struct BBox {
    int x = 0, y = 0, w = 0, h = 0;
    bool valid() const { return w > 0 && h > 0; }
};

class BboxU2netp {
public:
    struct Config {
        int inputSize = 320; // 网络输入边长
        int numThreads = 4; // 推理线程数
        bool useVulkan = true; // Android 上开 Vulkan
        bool useFP16 = true; // 存储/计算用 FP16
        std::string inputName = "in0"; // Netron 里看到的输入名
        std::string outputName = "out0"; // Netron 里看到的输出名
        bool applySigmoid = false; // 输出已是 0~1 时 false
        bool binaryMask = false; // false = 软 mask（推荐），true = 硬二值
        float threshold = 0.2f; // binaryMask=true 时生效
        float featherSigma = 1.2f; // 边缘羽化，0 = 关闭

        int minArea = 64; // 矩形框最小面积
        float expandRatio = 0.f;
    };

    BboxU2netp();
    explicit BboxU2netp(const Config& cfg);
    ~BboxU2netp();

    BboxU2netp(const BboxU2netp&) = delete;
    BboxU2netp& operator=(const BboxU2netp&) = delete;

    std::pair<cv::Mat, cv::Mat> u2netpCropROI(const cv::Mat& fg, double minArea = 1500, bool rotation = true);

    bool load_model(const std::string& paramPath,
        const std::string& binPath);

    const Config& config() const { return cfg_; }
    void setConfig(const Config& cfg) { cfg_ = cfg; }

private:
    bool isLoaded() const;

    // 输入 BGR CV_8UC3，返回与原图同尺寸的 CV_32FC1 mask（0~1）
    cv::Mat segment(const cv::Mat& bgr);

    // 分割 + 合成到纯色背景，返回 CV_8UC3
    cv::Mat process(const cv::Mat& bgr, const cv::Scalar& bg);

    // 分割 + 输出带 alpha 通道的 BGRA，返回 CV_8UC4
    cv::Mat processWithAlpha(const cv::Mat& bgr);

    // 用外部 mask 合成，mask 为 0~1 的 CV_32FC1
    static cv::Mat compose(const cv::Mat& bgr,
        const cv::Mat& mask,
        const cv::Scalar& bg);

private:
    cv::Mat runNetwork(const cv::Mat& bgr);
    cv::Mat postprocess(const cv::Mat& raw, const cv::Size& target);

    Config cfg_;
    ncnn::Net net_;
    bool loaded_ = false;
    std::mutex mtx_;
};

#endif // BBOXU2NETP_H
