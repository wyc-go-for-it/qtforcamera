#include "isnetsegmenter.h"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>

ISNetSegmenter::ISNetSegmenter()
{
    // 针对移动端优化：开启多线程
    // ncnn::set_omp_num_threads(4);
}

ISNetSegmenter::~ISNetSegmenter()
{
    net.clear();
}

bool ISNetSegmenter::load_model(const std::string& param_path, const std::string& bin_path)
{

    // 1. 设置网络选项
    net.opt.lightmode = true; // 开启轻量模式（移动端必选）
    net.opt.num_threads = 4; // 设置工作线程数

    // 针对移动端优化：开启 FP16 推理和算子融合（前提是模型是 fp16 的）
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;

    // 2. 加载模型
    if (net.load_param(param_path.c_str()) != 0) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        std::cerr << "Failed to load bin: " << bin_path << std::endl;
        return false;
    }
    std::cout << "IS-Net-Lite (FP16) model loaded successfully!" << std::endl;
    return true;
    return true;
}

std::pair<cv::Mat, cv::Mat> ISNetSegmenter::process(const cv::Mat& input_image, bool rotation)
{
    if (input_image.empty())
        return { cv::Mat(), cv::Mat() };

    cv::Mat fg3ch = input_image;
    if (input_image.channels() == 4)
        cv::cvtColor(input_image, fg3ch, cv::COLOR_BGRA2BGR);
    else if (input_image.channels() == 1)
        cv::cvtColor(input_image, fg3ch, cv::COLOR_GRAY2BGR);

    auto t1 = std::chrono::high_resolution_clock::now();

    const int img_w = fg3ch.cols;
    const int img_h = fg3ch.rows;
    const int target_size = 348;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        fg3ch.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img_w, img_h,
        target_size, target_size);

    // 标准化参数： pixel / 255.0 - 0.5
    const float mean_vals[3] = { 0.0f, 0.0f, 0.0f };
    const float norm_vals[3] = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 2. 推理
    ncnn::Extractor ex = net.create_extractor();

    // ！！！核心：请根据你的 .param 文件，将 "in0" 和 "out0"
    // 替换为 Input 和 Output Layer 的实际 top/bottom 名称。
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    // 3. 后处理（关键修复区 & 移动端零拷贝优化）

    // --- 移动端优化点 ---
    // 直接创建一个 OpenCV Mat，它共享 ncnn::Mat 'out' 的内存数据。
    // 这避免了模型输出数据的再次拷贝，极大降低了能耗和内存带宽开销。
    cv::Mat mask_fp32(target_size, target_size, CV_32FC1, (void*)out.data);

    // 1) 修复：手动应用 Sigmoid 激活。
    // 将数值从 [-inf, +inf] 压缩到 [0, 1] 之间。
    apply_sigmoid_in_place(mask_fp32);

    // 2) 数据转换：[0, 1] (Float) -> [0, 255] (UChar)
    // 并将结果直接写入最终的 alpha_mask。
    cv::Mat mask_u8;
    mask_fp32.convertTo(mask_u8, CV_8UC1, 255.0);

    cv::Mat alpha_mask;
    // 3) 将 mask 缩放到与原图一致的尺寸。
    // 为了速度，推荐先将模型输出尺寸缩放，再转 U8。
    cv::resize(mask_u8, alpha_mask, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat productMask;
    cv::threshold(alpha_mask, productMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "推理:" << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(productMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
        return { cv::Mat(), cv::Mat() };

    std::vector<cv::Point> allPts;
    for (const auto& c : contours) {
        if (cv::contourArea(c) > 1500) {
            allPts.insert(allPts.end(), c.begin(), c.end());
        }
    }

    cv::rectangle(input_image,
        cv::boundingRect(allPts) & cv::Rect(0, 0, input_image.cols, input_image.rows),
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
