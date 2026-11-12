#include "SilentFaceAntiSpoofing.h"
#include <algorithm>
#include <cmath>

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

SilentFaceAntiSpoofing::SilentFaceAntiSpoofing() { }
SilentFaceAntiSpoofing::~SilentFaceAntiSpoofing() { }

bool SilentFaceAntiSpoofing::load_model(const std::string& param_path, const std::string& bin_path)
{
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 2;
    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;

    if (net.load_param(param_path.c_str()) != 0) {
        printf("AntiSpoofing load_param failed\n");
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        printf("AntiSpoofing load_model failed\n");
        return false;
    }
    return true;
}

AntiSpoofingResult SilentFaceAntiSpoofing::detect(const cv::Mat& img, const cv::Rect& face_rect)
{
    AntiSpoofingResult res = { false, 0.f, 0.f, 0.f };

    float scales[] = { 1.35, 2.7, 4.0 };
    float total_real = 0.f, total_print = 0.f, total_screen = 0.f;

    for (size_t i = 0; i < 3; i++) {

        float cx = face_rect.x + face_rect.width * 0.5f;
        float cy = face_rect.y + face_rect.height * 0.5f;
        float side = std::max(face_rect.width, face_rect.height) * scales[i];
        float half_side = side * 0.5f;

        cv::Rect2f crop_rect(cx - half_side, cy - half_side, side, side);

        crop_rect &= cv::Rect2f(0, 0, img.cols, img.rows);

        if (crop_rect.width <= 0 || crop_rect.height <= 0) {
            return res;
        }

        cv::Mat crop = img(cv::Rect(crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height)).clone();
        cv::Mat resized_80;
        cv::resize(crop, resized_80, cv::Size(input_size, input_size));

        ncnn::Mat in = ncnn::Mat::from_pixels(resized_80.data, ncnn::Mat::PIXEL_BGR, input_size, input_size);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        if (ex.extract("out0", out) != 0) {
            LOGI("AntiSpoofing extract failed\n");
            return res;
        }

        if (out.w != 3) {
            LOGI("活体模型输出维度不是3: w=%d\n", out.w);
            return res;
        }

        float exp_sum = 0.f;
        float probs[3];
        for (int i = 0; i < 3; ++i) {
            probs[i] = std::exp(out[i]);
            exp_sum += probs[i];
        }
        for (int i = 0; i < 3; ++i) {
            probs[i] /= exp_sum;
        }

        total_real += probs[1];
        total_print += probs[0];
        total_screen += probs[2];
    }

    float final_real = total_real / 3.0f;
    float final_screen = total_screen / 3.0f;
    float final_print = total_print / 3.0f;

    res.real_score = final_real;
    res.print_score = final_print;
    res.screen_score = final_screen;

    res.is_real = (res.real_score > 0.85f);

    LOGI("活体检测: 真人=%.3f, 打印=%.3f, 屏幕=%.3f -> %s\n",
        res.real_score, res.print_score, res.screen_score,
        res.is_real ? "真脸" : "假体");

    return res;
}
