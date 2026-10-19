#include <ncnn/net.h>
#include <opencv2/core.hpp>

class ISNetSegmenter {
public:
    ISNetSegmenter();

    ~ISNetSegmenter();

    // 初始化模型
    bool load_model(const std::string& param_path, const std::string& bin_path);

    // 执行抠图推理
    // input_image: 原始输入图片 (BGR 格式)
    // alpha_mask: 输出的单通道透明度掩码 (0-255)
    std::pair<cv::Mat, cv::Mat> process(const cv::Mat& input_image, bool rotation = true);

private:
    // 手动 Sigmoid 激活函数实现
    static float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // 就地（In-Place）应用 Sigmoid 激活到 cv::Mat
    static void apply_sigmoid_in_place(cv::Mat& mat)
    {
        if (mat.isContinuous()) {
            float* ptr = mat.ptr<float>(0);
            int size = mat.cols * mat.rows;
            for (int i = 0; i < size; ++i) {
                ptr[i] = sigmoid(ptr[i]);
            }
        } else {
            // 处理非连续内存的情况（通常很少见）
            for (int r = 0; r < mat.rows; ++r) {
                float* ptr = mat.ptr<float>(r);
                for (int c = 0; c < mat.cols; ++c) {
                    ptr[c] = sigmoid(ptr[c]);
                }
            }
        }
    }

private:
    ncnn::Net net;
};
