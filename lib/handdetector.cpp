#include "handdetector.h"
#include "ncnn/cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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

HandDetector::HandDetector()
{
}

HandDetector::~HandDetector()
{
    net.clear();
}

bool HandDetector::load_model(const std::string param, const std::string bin)
{
#if defined(__ANDROID__)
    net.opt.num_threads = ncnn::get_big_cpu_count();

    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
#else
    net.opt.num_threads = 4;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
#endif

    net.opt.blob_allocator = 0;
    net.opt.workspace_allocator = 0;
    net.opt.openmp_blocktime = 0;

    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;

    net.opt.lightmode = true;

    if (net.load_param(param.c_str()) != 0) {
        LOGE("[HandDetector] load_param failed: %s\n", param.c_str());
        return false;
    }
    if (net.load_model(bin.c_str()) != 0) {
        LOGE("[HandDetector] load_model failed: %s\n", bin.c_str());
        return false;
    }
    loaded = true;

    LOGI("[HandDetector] has loaded\n");

    return true;
}

cv::Mat letterbox(const cv::Mat& src, int target_size, ScaleParameter& parameter,
    cv::Scalar color = cv::Scalar(114, 114, 114))
{
    int w = src.cols, h = src.rows;
    float scale = std::min((float)target_size / w, (float)target_size / h);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_w = (target_size - new_w) / 2;
    int pad_h = (target_size - new_h) / 2;

    cv::Mat out;
    cv::copyMakeBorder(resized, out,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv::BORDER_CONSTANT, color);

    LOGI("letterbox: scale=%.3f pad_w=%d pad_h=%d\n", scale, pad_w, pad_h);

    parameter.scale = scale;
    parameter.pad_x = pad_w;
    parameter.pad_y = pad_h;

    return out;
}

std::vector<Detection> HandDetector::detect(const cv::Mat& bgr, float conf_thresh, float iou_thresh)
{
    std::vector<Detection> results;
    if (!loaded || bgr.empty())
        return results;

    auto t0 = std::chrono::high_resolution_clock::now();

    const int input_size = 320;

    ScaleParameter scaleParam;
    cv::Mat input = letterbox(bgr, input_size, scaleParam);

    float pre_scale = scaleParam.scale;
    int pad_x = scaleParam.pad_x;
    int pad_y = scaleParam.pad_y;

    ncnn::Mat in = ncnn::Mat::from_pixels(
        input.data, ncnn::Mat::PIXEL_BGR2RGB,
        input_size, input_size);

    const float mean[3] = { 0.f, 0.f, 0.f };
    const float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in);

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
        return results;
    if (out.w != 2100 || out.h != 5) {
        LOGE("unexpected out: w=%d h=%d\n", out.w, out.h);
        return results;
    }

    const float* cx_ptr = out.row(0);
    const float* cy_ptr = out.row(1);
    const float* w_ptr = out.row(2);
    const float* h_ptr = out.row(3);
    const float* conf_ptr = out.row(4);

    std::vector<std::pair<float, int>> confs;

    for (int i = 0; i < 2100; ++i) {
        float conf = conf_ptr[i];

        confs.push_back({ conf, i });

        if (conf < conf_thresh)
            continue;

        float cx = cx_ptr[i];
        float cy = cy_ptr[i];
        float w = w_ptr[i];
        float h = h_ptr[i];

        float x1 = (cx - w * 0.5f - pad_x) / pre_scale;
        float y1 = (cy - h * 0.5f - pad_y) / pre_scale;
        float x2 = (cx + w * 0.5f - pad_x) / pre_scale;
        float y2 = (cy + h * 0.5f - pad_y) / pre_scale;

        x1 = std::max(0.f, std::min((float)bgr.cols, x1));
        y1 = std::max(0.f, std::min((float)bgr.rows, y1));
        x2 = std::max(0.f, std::min((float)bgr.cols, x2));
        y2 = std::max(0.f, std::min((float)bgr.rows, y2));

        if (x2 - x1 < 3.f || y2 - y1 < 3.f)
            continue;

        results.push_back({ cv::Rect_<float> { x1, y1, x2 - x1, y2 - y1 }, conf });
    }

    std::sort(confs.begin(), confs.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    nmsInPlace(results, iou_thresh);

    LOGI("【耗时分析】HandDetector: %.2f ms\n", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count());

    return results;
}

void HandDetector::label(const std::string& img_dir, const std::string& label_dir, const std::string& vis_dir, uint32_t& num_files)
{

    if (!loaded) {
        return;
    }

    namespace fs = std::filesystem;

    auto conf_thresh = 0.5f;
    auto iou_thresh = 0.6;

    fs::create_directories(label_dir);
    if (!vis_dir.empty())
        fs::create_directories(vis_dir);

    std::vector<std::string> exts = { ".jpg", ".jpeg", ".png", ".bmp" };
    int processed = 0, skipped = 0, total_boxes = 0;

    fs::path last_path;

    for (auto& entry : fs::directory_iterator(img_dir)) {
        if (!entry.is_regular_file())
            continue;

        num_files++;

        std::string path = entry.path().string();

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end())
            continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            LOGI("read failed: %s\n", path.c_str());
            skipped++;
            continue;
        }

        std::vector<Detection> dets = detect(img, conf_thresh, iou_thresh);

        if (dets.empty()) {
            // fs::remove(path);
            LOGI("删除文件：%s\n", path.c_str());
            continue;
        }

        last_path = entry.path();

        std::string stem = entry.path().stem().string();

        // 写 YOLO 标签
        bool txt = false;
        if (txt) {
            std::string lbl_path = label_dir + "/" + stem + ".txt";
            std::ofstream fp(lbl_path);
            if (!fp.is_open()) {
                LOGI("open label failed: %s\n", lbl_path.c_str());
                skipped++;
                continue;
            }
            int w = img.cols, h = img.rows;
            for (auto& d : dets) {
                auto rect = d.rect; // cv::Rect_<float>: x, y, width, height

                // YOLO: cx, cy 是中心点，bw, bh 是宽高，全部归一化到 [0,1]
                float cx = (rect.x + rect.width * 0.5f) / w;
                float cy = (rect.y + rect.height * 0.5f) / h;
                float bw = rect.width / w;
                float bh = rect.height / h;

                // clamp 到 [0,1]，防止超界
                cx = std::max(0.f, std::min(1.f, cx));
                cy = std::max(0.f, std::min(1.f, cy));
                bw = std::max(0.f, std::min(1.f, bw));
                bh = std::max(0.f, std::min(1.f, bh));

                fp << "0 " << cx << " " << cy << " " << bw << " " << bh << "\n";
            }
            fp.close();
        } else {

            std::string json_path = label_dir + "/" + stem + ".json";
            std::ofstream jf(json_path);
            if (!jf.is_open()) {
                LOGI("open json failed: %s\n", json_path.c_str());
                skipped++;
                continue;
            }

            // 获取带后缀的图片名（用于 imagePath 字段）
            std::string img_name = entry.path().filename().string();

            // 开始拼接 JSON
            jf << "{\n";
            jf << "  \"version\": \"4.2.0\",\n";
            jf << "  \"flags\": {},\n";
            jf << "  \"checked\": false,\n";
            jf << "  \"shapes\": [\n";

            int w = img.cols, h = img.rows;
            jf << std::fixed << std::setprecision(6); // 设置浮点数精度

            for (size_t i = 0; i < dets.size(); ++i) {
                auto rect = dets[i].rect; // cv::Rect_<float>: x, y, width, height

                // 计算 4 个角点，并限制在图片范围内
                float x1 = std::max(0.f, std::min((float)w, rect.x));
                float y1 = std::max(0.f, std::min((float)h, rect.y));
                float x2 = std::max(0.f, std::min((float)w, rect.x + rect.width));
                float y2 = std::max(0.f, std::min((float)h, rect.y + rect.height));

                jf << "    {\n";
                jf << "      \"label\": \"hand\",\n";
                // 如果想把置信度填进去，把 null 改成 dets[i].prob
                jf << "      \"score\": null,\n";
                jf << "      \"points\": [\n";
                jf << "        [" << x1 << ", " << y1 << "],\n"; // 左上
                jf << "        [" << x2 << ", " << y1 << "],\n"; // 右上
                jf << "        [" << x2 << ", " << y2 << "],\n"; // 右下
                jf << "        [" << x1 << ", " << y2 << "]\n"; // 左下
                jf << "      ],\n";
                jf << "      \"group_id\": null,\n";
                jf << "      \"description\": \"\",\n";
                jf << "      \"difficult\": false,\n";
                jf << "      \"shape_type\": \"rectangle\",\n";
                jf << "      \"flags\": {},\n";
                jf << "      \"attributes\": {},\n";
                jf << "      \"kie_linking\": []\n";
                jf << "    }";

                // 不是最后一个框，加逗号
                if (i < dets.size() - 1)
                    jf << ",";
                jf << "\n";
            }

            jf << "  ],\n";
            jf << "  \"imagePath\": \"" << img_name << "\",\n";
            jf << "  \"imageData\": null,\n";
            jf << "  \"imageHeight\": " << h << ",\n";
            jf << "  \"imageWidth\": " << w << "\n";
            jf << "}\n";

            jf.close();

            cv::imwrite(label_dir + "/" + stem + ".jpg", img);
        }

        // 可视化
        if (!vis_dir.empty()) {
            cv::Mat vis = img;
            for (auto& d : dets) {
                auto rect = d.rect;

                cv::rectangle(vis, rect,
                    cv::Scalar(0, 255, 0), 2);
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", d.prob);
                cv::putText(vis, buf,
                    cv::Point((int)rect.x, std::max(12, (int)rect.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
            }
            cv::imwrite(vis_dir + "/" + stem + ".jpg", vis);
        }

        processed++;
        total_boxes += (int)dets.size();
        if (processed % 150 == 0) {
            LOGI("[%d] %s: %zu hands\n", processed, stem.c_str(), dets.size());
        }
    }

    LOGI("\n=== done ===\n");
    LOGI("processed: %d, skipped: %d\n", processed, skipped);
    LOGI("total boxes: %d, avg: %.2f\n",
        total_boxes, processed > 0 ? (float)total_boxes / processed : 0.f);

    if (!last_path.empty()) {

        cv::Mat img = cv::imread(last_path.string());
        int w = img.cols, h = img.rows;

        std::ifstream f("label_dir/" + last_path.stem().string() + ".txt");
        std::string line;
        while (std::getline(f, line)) {
            float cls, cx, cy, bw, bh;
            if (sscanf(line.c_str(), "%f %f %f %f %f", &cls, &cx, &cy, &bw, &bh) == 5) {
                float x1 = (cx - bw / 2) * w;
                float y1 = (cy - bh / 2) * h;
                float x2 = (cx + bw / 2) * w;
                float y2 = (cy + bh / 2) * h;
                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2); // 红色
            }
        }
        cv::imwrite("check_label.jpg", img);
    }
}

void HandDetector::nms(std::vector<Detection>& dets, float iou_thresh)
{
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.prob > b.prob;
        });
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i])
            continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j])
                continue;

            const auto& i_rect = dets[i].rect;
            const auto& j_rect = dets[j].rect;

            float inter = std::max(0.0f, std::max(i_rect.area(), j_rect.area()));
            float a = dets[i].rect.area();
            float b = dets[j].rect.area();
            if (inter / (a + b - inter + 1e-6f) > iou_thresh) {
                removed[j] = true;
            }
        }
    }
    std::vector<Detection> keep;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!removed[i])
            keep.push_back(dets[i]);
    }
    dets = keep;
}

float HandDetector::iou(const Detection& a, const Detection& b)
{
    const auto& i_rect = a.rect;
    const auto& j_rect = b.rect;

    // 交集
    float inter_x1 = std::max(i_rect.x, j_rect.x);
    float inter_y1 = std::max(i_rect.y, j_rect.y);
    float inter_x2 = std::min(i_rect.x + i_rect.width, j_rect.x + j_rect.width);
    float inter_y2 = std::min(i_rect.y + i_rect.height, j_rect.y + j_rect.height);

    float inter_w = std::max(0.f, inter_x2 - inter_x1);
    float inter_h = std::max(0.f, inter_y2 - inter_y1);
    float inter = inter_w * inter_h;

    if (inter <= 0.f)
        return 0.f;

    // 并集
    float area_a = i_rect.area();
    float area_b = j_rect.area();
    float uni = area_a + area_b - inter;

    if (uni <= 0.f)
        return 0.f;

    return inter / uni;
}

void HandDetector::nmsInPlace(std::vector<Detection>& dets, float iou_thresh)
{
    if (dets.empty())
        return;

    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.prob > b.prob;
        });

    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i])
            continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j])
                continue;
            if (iou(dets[i], dets[j]) > iou_thresh) {
                removed[j] = true;
            }
        }
    }

    std::vector<Detection> keep;
    keep.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!removed[i])
            keep.push_back(dets[i]);
    }
    dets.swap(keep);
}
