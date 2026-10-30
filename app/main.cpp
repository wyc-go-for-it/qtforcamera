#include "camera.h"
#include "qcoreapplication.h"

#include <QApplication>

#ifdef Q_OS_ANDROID
#include <androidUtils/localandroidutils.h>
#endif

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

void splitData()
{

    namespace fs = std::filesystem;

    // ================= 配置区 =================
    std::string src_dir_path = "labels"; // 你的源文件夹
    std::string val_dir_path = "labels/val"; // 目标文件夹
    double val_ratio = 0.1; // 抽取比例 10%
    // ==========================================

    // 1. 检查源文件夹是否存在
    if (!fs::exists(src_dir_path) || !fs::is_directory(src_dir_path)) {
        std::cerr << "错误: 源目录不存在: " << src_dir_path << std::endl;
        return;
    }

    // 2. 创建 val 目录
    if (!fs::exists(val_dir_path)) {
        fs::create_directories(val_dir_path);
    }

    // 3. 收集所有图片文件
    std::vector<std::string> img_files;
    for (const auto& entry : fs::directory_iterator(src_dir_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            // 匹配常见的图片格式
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                img_files.push_back(entry.path().filename().string());
            }
        }
    }

    if (img_files.empty()) {
        std::cout << "源目录中没有找到图片文件。" << std::endl;
        return;
    }

    // 4. 随机打乱
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(img_files.begin(), img_files.end(), std::default_random_engine(seed));

    // 5. 计算抽取数量
    size_t total_count = img_files.size();
    size_t val_count = static_cast<size_t>(total_count * val_ratio);

    std::cout << "总图片数: " << total_count << std::endl;
    std::cout << "计划抽取: " << val_count << " 张到 val 文件夹" << std::endl;

    // 6. 成对拷贝图片和对应的 txt
    size_t copied_img = 0;
    size_t copied_txt = 0;
    size_t missing_txt = 0;

    for (size_t i = 0; i < val_count; ++i) {
        std::string img_name = img_files[i];
        std::string stem = fs::path(img_name).stem().string(); // 获取不带后缀的文件名
        std::string txt_name = stem + ".txt";

        fs::path src_img = fs::path(src_dir_path) / img_name;
        fs::path src_txt = fs::path(src_dir_path) / txt_name;
        fs::path dst_img = fs::path(val_dir_path) / img_name;
        fs::path dst_txt = fs::path(val_dir_path) / txt_name;

        try {
            // 拷贝图片
            fs::rename(src_img, dst_img);
            copied_img++;

            // 拷贝标签（如果存在）
            if (fs::exists(src_txt)) {
                fs::rename(src_txt, dst_txt);
                copied_txt++;
            } else {
                missing_txt++;
                std::cout << "警告: " << img_name << " 没有找到对应的 txt 标签" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "拷贝失败: " << img_name << " - " << e.what() << std::endl;
        }
    }

    std::cout << "\n=== 拷贝完成 ===" << std::endl;
    std::cout << "成功拷贝图片: " << copied_img << " 张" << std::endl;
    std::cout << "成功拷贝标签: " << copied_txt << " 个" << std::endl;
    if (missing_txt > 0) {
        std::cout << "缺失标签: " << missing_txt << " 个" << std::endl;
    }
}

int main(int argc, char* argv[])
{

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0))
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QCoreApplication::setAttribute(Qt::AA_NativeWindows);

    QApplication app(argc, argv);

    qRegisterMetaType<QVector<QPointF>>();
    qRegisterMetaType<QList<RecogResult>>();
    qRegisterMetaType<QVector<DetectObject>>();

#ifdef Q_OS_ANDROID
    if (!LocalAndroidUtils::checkAndRequestCameraPermission()) {
        return 1;
    }
#endif

#ifdef _WIN32
    SetConsoleOutputCP(65001); // 65001即 CP_UTF8
    SetConsoleCP(65001);
#endif

    Camera camera;
    camera.show();

    // splitData();

    return app.exec();
};
