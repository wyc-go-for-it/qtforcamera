#ifndef LITESALIENCYNET_H
#define LITESALIENCYNET_H

#include <string>

#include <ncnn/net.h>

#include <opencv2/core/mat.hpp>

#include "mylib/mylib_export.h"

class WYCAI_EXPORT LiteSaliencyNet {
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

    LiteSaliencyNet();
    ~LiteSaliencyNet();

    bool load_model(const std::string& paramPath,
        const std::string& binPath);

    std::pair<cv::Mat, cv::Mat> liteSaliencyCropROI(const cv::Mat& fg, double minArea = 1500, bool rotation = true);

    const Config& config() const { return m_cfg; }
    void setConfig(const Config& cfg) { m_cfg = cfg; }

private:
    cv::Mat runNetwork(const cv::Mat& bgr);
    cv::Mat createMask(const cv::Mat& bgr);

private:
    Config m_cfg;
    ncnn::Net m_net;
    bool m_loaded = false;
};

#endif // LITESALIENCYNET_H
