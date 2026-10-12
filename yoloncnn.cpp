#include "yoloncnn.h"
#include "ncnn/cpu.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

Yolo11Ncnn::Yolo11Ncnn()
    : target_size(640)
    , is_loaded(false)
{
}

Yolo11Ncnn::~Yolo11Ncnn()
{
    net.clear();
}

bool Yolo11Ncnn::load_model(const std::string& param_path,
    const std::string& bin_path,
    int target_size,
    bool use_gpu)
{
    this->target_size = target_size;

    net.opt.use_vulkan_compute = use_gpu;

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

    if (net.load_param(param_path.c_str()) != 0)
        return false;
    if (net.load_model(bin_path.c_str()) != 0)
        return false;

    is_loaded = true;

    return true;
}

ncnn::Mat Yolo11Ncnn::
    preprocess_letterbox(const cv::Mat& bgr, LetterboxParams& params)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    float scale = std::min((float)target_size / img_w, (float)target_size / img_h);
    params.scale = scale;

    int new_w = (int)std::round(img_w * scale);
    int new_h = (int)std::round(img_h * scale);

    int w_pad = target_size - new_w;
    int h_pad = target_size - new_h;

    params.pad_w = w_pad / 2;
    params.pad_h = h_pad / 2;

    int top = params.pad_h;
    int bottom = h_pad - params.pad_h;
    int left = params.pad_w;
    int right = w_pad - params.pad_w;

    ncnn::Mat in_resized = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
        img_w, img_h,
        new_w, new_h);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_resized.substract_mean_normalize(0, norm_vals);

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in_resized, in_pad,
        top, bottom, left, right,
        ncnn::BORDER_CONSTANT, 114.f / 255.f);

    return in_pad;
}

static inline float decode_dfl(const float* ptr)
{
    float sum = 0.0f;
    float exp_vals[16];
    float max_val = ptr[0];
    for (int i = 1; i < 16; i++) {
        if (ptr[i] > max_val)
            max_val = ptr[i];
    }

    for (int i = 0; i < 16; i++) {
        exp_vals[i] = std::exp(ptr[i] - max_val);
        sum += exp_vals[i];
    }

    float dfl_val = 0.0f;
    for (int i = 0; i < 16; i++) {
        float prob = exp_vals[i] / sum;
        dfl_val += i * prob;
    }
    return dfl_val;
}

bool Yolo11Ncnn::detect(const cv::Mat& bgr,
    std::vector<Object>& objects,
    float conf_thresh,
    float nms_thresh)
{
    return detect_modelB(bgr, objects, conf_thresh, nms_thresh);
}

inline float Yolo11Ncnn::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

bool Yolo11Ncnn::detect_modelA(const cv::Mat& bgr, std::vector<Object>& objects, float conf_thresh, float nms_thresh)
{
    if (!is_loaded || bgr.empty())
        return false;

    objects.clear();

    auto t0 = std::chrono::high_resolution_clock::now();

    LetterboxParams params;
    ncnn::Mat in_mat = preprocess_letterbox(bgr, params);

    auto t1 = std::chrono::high_resolution_clock::now();

    ncnn::Extractor ex = net.create_extractor();

    if (ex.input("in0", in_mat) != 0)
        return false;

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
        return false;

    auto t2 = std::chrono::high_resolution_clock::now();

    int num_anchors = out.h;
    int num_attributes = out.w;

    const int reg_max = 16;
    const int num_box_channels = 4 * reg_max; // 64
    int num_classes = num_attributes - num_box_channels;

    std::vector<Object> proposal_objects;

    static const int strides[3] = { 8, 16, 32 };
    // static const int grid_sizes[3] = { 80, 40, 20 }; //{80, 40, 20} 是严格为 640x640 计算的特征图网格大小（$640/8=80$, $640/16=40$, $640/32=20$）。

    int anchor_idx = 0;

    for (int s = 0; s < 3; s++) {
        int stride = strides[s];
        int grid_size = this->target_size / stride;

        for (int g_y = 0; g_y < grid_size; g_y++) {
            for (int g_x = 0; g_x < grid_size; g_x++) {
                if (anchor_idx >= num_anchors)
                    break;

                // 直接提取第 anchor_idx 行的数据（长 144 的数组）
                const float* feat = out.row(anchor_idx);

                // 1. 寻找 80 个类别中的最大 Logit
                int label = -1;
                float max_logit = -1000.0f;
                for (int c = 0; c < num_classes; c++) {
                    float logit = feat[64 + c];
                    if (logit > max_logit) {
                        max_logit = logit;
                        label = c;
                    }
                }

                // 2. 计算 Sigmoid 置信度
                float score = 1.0f / (1.0f + std::exp(-max_logit));

                if (score >= conf_thresh) {
                    // 3. 计算网格中心点坐标（单位：640像素）
                    float cx = (g_x + 0.5f) * stride;
                    float cy = (g_y + 0.5f) * stride;

                    // 4. 解码 DFL 距离并乘以 stride
                    float d_left = decode_dfl(feat + 0) * stride;
                    float d_top = decode_dfl(feat + 16) * stride;
                    float d_right = decode_dfl(feat + 32) * stride;
                    float d_bottom = decode_dfl(feat + 48) * stride;

                    // 5. 还原到 640x640 下的 bounding box (x1, y1, x2, y2)
                    float x1_640 = cx - d_left;
                    float y1_640 = cy - d_top;
                    float x2_640 = cx + d_right;
                    float y2_640 = cy + d_bottom;

                    // 6. 根据 Letterbox 参数还原回原图坐标
                    float x0 = (x1_640 - params.pad_w) / params.scale;
                    float y0 = (y1_640 - params.pad_h) / params.scale;
                    float x1 = (x2_640 - params.pad_w) / params.scale;
                    float y1 = (y2_640 - params.pad_h) / params.scale;

                    // 边界裁切 Clamp
                    x0 = std::max(0.0f, std::min(x0, (float)bgr.cols - 1.0f));
                    y0 = std::max(0.0f, std::min(y0, (float)bgr.rows - 1.0f));
                    x1 = std::max(0.0f, std::min(x1, (float)bgr.cols - 1.0f));
                    y1 = std::max(0.0f, std::min(y1, (float)bgr.rows - 1.0f));

                    float w = x1 - x0;
                    float h = y1 - y0;

                    if (w > 2.0f && h > 2.0f) {
                        Object obj;
                        obj.rect = cv::Rect_<float>(x0, y0, w, h);
                        obj.label = label;
                        obj.prob = score;
                        proposal_objects.push_back(obj);
                    }
                }

                anchor_idx++;
            }
        }
    }

    // 按置信度排序并做 NMS
    std::sort(proposal_objects.begin(), proposal_objects.end(),
        [](const Object& a, const Object& b) { return a.prob > b.prob; });

    nms_sorted_bboxes(proposal_objects, objects, nms_thresh);

    auto t3 = std::chrono::high_resolution_clock::now();

    double time_prep = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double time_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double time_post = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double time_total = std::chrono::duration<double, std::milli>(t3 - t0).count();

#if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_INFO, "qtforcamera", "in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,num_classes:%d\n", in_mat.w, in_mat.h, in_mat.c, out.h, out.w, num_classes);

    __android_log_print(ANDROID_LOG_INFO, "qtforcamera",
        "【耗时分析】预处理: %.2f ms | 推理: %.2f ms | 后处理: %.2f ms | 总计: %.2f ms",
        time_prep, time_infer, time_post, time_total);
#else
    fprintf(stderr, "in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,num_classes:%d\n", in_mat.w, in_mat.h, in_mat.c, out.h, out.w, num_classes);

    printf("【耗时分析】预处理: %.2f ms | 推理: %.2f ms | 后处理: %.2f ms | 总计: %.2f ms\n",
        time_prep, time_infer, time_post, time_total);

    fflush(stdout);
#endif

    return true;
}

bool Yolo11Ncnn::detect_modelB(const cv::Mat& bgr, std::vector<Object>& objects, float conf_thresh, float nms_thresh)
{
    if (!is_loaded || bgr.empty())
        return false;

    objects.clear();

    auto t0 = std::chrono::high_resolution_clock::now();

    // 模型 B 是 640 导出，target_size 必须是 640
    LetterboxParams params;
    ncnn::Mat in_mat = preprocess_letterbox(bgr, params);

    auto t1 = std::chrono::high_resolution_clock::now();

    ncnn::Extractor ex = net.create_extractor();
    ex.set_blob_allocator(0);
    ex.set_workspace_allocator(0);
    ex.set_light_mode(true);

    if (ex.input("in0", in_mat) != 0)
        return false;

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
        return false;

    auto t2 = std::chrono::high_resolution_clock::now();

    // 关键：维度是 out.h = 属性数(84)，out.w = 锚点数(8400)
    int num_attributes = out.h; // 84
    int num_anchors = out.w; // 8400
    int num_classes = num_attributes - 4; // 80

    // 预先取出每一行的指针，避免内层循环反复调用 row()
    std::vector<const float*> rows(num_attributes);
    for (int r = 0; r < num_attributes; r++)
        rows[r] = out.row(r);

    std::vector<Object> proposal_objects;
    proposal_objects.reserve(256);

    for (int j = 0; j < num_anchors; j++) {
        // 找最大类别分数（网络里已经 Sigmoid，不要再做）
        int label = -1;
        float score = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float s = rows[4 + c][j];
            if (s > score) {
                score = s;
                label = c;
            }
        }

        if (score < conf_thresh)
            continue;

        // box：模型 B 输出 xywh（中心点 + 宽高，640 尺度）
        float cx = rows[0][j];
        float cy = rows[1][j];
        float w = rows[2][j];
        float h = rows[3][j];

        float x1_640 = cx - w * 0.5f;
        float y1_640 = cy - h * 0.5f;
        float x2_640 = cx + w * 0.5f;
        float y2_640 = cy + h * 0.5f;

        // 还原回原图坐标
        float x0 = (x1_640 - params.pad_w) / params.scale;
        float y0 = (y1_640 - params.pad_h) / params.scale;
        float x1 = (x2_640 - params.pad_w) / params.scale;
        float y1 = (y2_640 - params.pad_h) / params.scale;

        x0 = std::max(0.0f, std::min(x0, (float)bgr.cols - 1.0f));
        y0 = std::max(0.0f, std::min(y0, (float)bgr.rows - 1.0f));
        x1 = std::max(0.0f, std::min(x1, (float)bgr.cols - 1.0f));
        y1 = std::max(0.0f, std::min(y1, (float)bgr.rows - 1.0f));

        float bw = x1 - x0;
        float bh = y1 - y0;
        if (bw <= 2.0f || bh <= 2.0f)
            continue;

        Object obj;
        obj.rect = cv::Rect_<float>(x0, y0, bw, bh);
        obj.label = label;
        obj.prob = score;
        proposal_objects.push_back(obj);
    }

    std::sort(proposal_objects.begin(), proposal_objects.end(),
        [](const Object& a, const Object& b) { return a.prob > b.prob; });

    nms_sorted_bboxes(proposal_objects, objects, nms_thresh);

    auto t3 = std::chrono::high_resolution_clock::now();

    double time_prep = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double time_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double time_post = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double time_total = std::chrono::duration<double, std::milli>(t3 - t0).count();

#if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_INFO, "qtforcamera", "in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,num_classes:%d\n", in_mat.w, in_mat.h, in_mat.c, out.h, out.w, num_classes);

    __android_log_print(ANDROID_LOG_INFO, "qtforcamera",
        "【耗时分析】预处理: %.2f ms | 推理: %.2f ms | 后处理: %.2f ms | 总计: %.2f ms",
        time_prep, time_infer, time_post, time_total);
#else
    fprintf(stderr, "in_mat: %d x %d x %d ,out.h: %d ,out.w: %d,num_classes:%d\n", in_mat.w, in_mat.h, in_mat.c, out.h, out.w, num_classes);

    printf("【耗时分析】预处理: %.2f ms | 推理: %.2f ms | 后处理: %.2f ms | 总计: %.2f ms\n",
        time_prep, time_infer, time_post, time_total);

    fflush(stdout);
#endif

    return true;
}

void Yolo11Ncnn::nms_sorted_bboxes(const std::vector<Object>& objects,
    std::vector<Object>& picked,
    float nms_thresh)
{
    picked.clear();
    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }

    for (size_t i = 0; i < n; i++) {
        const Object& a = objects[i];

        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++) {
            const Object& b = picked[j];

            if (a.label == b.label) {
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + b.rect.area() - inter_area;
                if (inter_area / union_area > nms_thresh) {
                    keep = false;
                    break;
                }
            }
        }

        if (keep) {
            picked.push_back(a);
        }
    }
}
