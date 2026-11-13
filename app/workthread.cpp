#include "workthread.h"
#include "camera.h"
#include "imageUtils.h"
#include "opencv2/imgcodecs.hpp"
#include "productfeatureengine.h"
#include "qdebug.h"

#include <opencv2/core/mat.hpp>

#include <BboxU2netp.h>
#include <ISNetSegmenter.h>
#include <OcrLite.h>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>
#include <handdetector.h>
#include <litesaliencynet.h>
#include <random>
#include <yoloncnn.h>

inline QDebug operator<<(QDebug dbg, const cv::Rect& rect)
{
    QDebugStateSaver saver(dbg);
    dbg.nospace() << "cv::Rect(x:" << rect.x
                  << ", y:" << rect.y
                  << ", w:" << rect.width
                  << ", h:" << rect.height << ")";
    return dbg;
}

inline QDebug operator<<(QDebug dbg, const cv::Point& pt)
{
    QDebugStateSaver saver(dbg);
    dbg.nospace() << "cv::Point(" << pt.x << ", " << pt.y << ")";
    return dbg;
}

WorkThread::WorkThread(QObject* parent)
    : QObject(parent)
{
    moveToThread(&m_thread);

    m_productDatabase.reset(new ProductDatabase(ProductFeatureEngine::featureDim));

    std::string path = WorkThread::ensureModelFile(":/models/models/mobilenet_v3_small_embedding.ncnn", "embedding.ncnn").toStdString();
    m_featureEngine.reset(new ProductFeatureEngine(path + ".param", path + ".bin"));

    connect(&m_thread, &QThread::started, this, [this]() {
        initDir();

        initOcrModel();

        initYoloModel();

        initU2netpModel();

        initLiteSaliencyNetModel();

        initYoloHandModel();

        m_productDatabase->loadModel((modelDir() + "/data.bat").toStdString());

        loadBG();
    });
}

WorkThread::~WorkThread()
{
    m_productDatabase->saveModel((modelDir() + "/data.bat").toStdString());

    m_thread.quit();
    m_thread.wait(3000);
}

void WorkThread::init()
{
    m_thread.start();
}

void WorkThread::studying(const ProductRecord& record)
{
    m_studiedRecord = record;
    isStudied.storeRelaxed(true);
}

void WorkThread::onVideoFrameChanged(const FrameData& frame)
{
    if (!frame.isValid())
        return;

    {
        QElapsedTimer t;
        t.start();

        auto tImage = FrameConverter::toImage(frame).copy();

        QMutexLocker l(&m_lock);
        m_curIndex = (m_curIndex + 1) & (FRAME_LST_SIZE - 1);
        m_frame_lst[m_curIndex] = tImage;
    }

    if (!m_hanlde_frame.exchange(true)) {
        QMetaObject::invokeMethod(this, "handleFrame", Qt::QueuedConnection);
    }
}

void WorkThread::onRecognition(bool recog)
{
    isRecognition.storeRelaxed(recog);
}

void WorkThread::onUpdateVertex(QVector<QPointF> points, QCamera::Position pos, qint32 rotationAngle)
{

    m_rotationAngle = rotationAngle;
    m_mirrored = pos;

    m_cropPoints.clear();
    m_cropPoints.append(points);

    std::vector<cv::Point> pts;
    pts.reserve(points.size());
    for (const QPointF& pt : qAsConst(m_cropPoints)) {
        pts.emplace_back(qRound(pt.x()), qRound(pt.y()));
    }

    m_cropRect = cv::boundingRect(pts);
}

void WorkThread::recognition(const cv::Mat& cur_fg)
{
    // if (m_bg.empty()) {
    //     emit error("背景无效");
    //     return;
    // }

    if (cur_fg.empty()) {
        emit error("识别图片转换错误");
        return;
    }

    QElapsedTimer t;
    t.start();
    // 2. 自动抠图获得纯净 ROI

    /*
    cv::Mat cropBg = m_bg(m_cropRect & cv::Rect(0, 0, m_bg.cols, m_bg.rows));
    auto [cropGoods, mask] = ProductFeatureEngine::cropROI(cropBg, cur_fg);*/

    // auto [cropGoods, mask] = m_u2netp->u2netpCropROI(cur_fg);
    auto [cropGoods, mask] = m_LiteSaliencyNet->liteSaliencyCropROI(cur_fg);

    qDebug() << "CropROI:" << t.elapsed();

    cv::Mat copy_bg = m_bg.clone();
    cv::rectangle(copy_bg,
        m_cropRect & cv::Rect(0, 0, m_bg.cols, m_bg.rows),
        cv::Scalar(0, 0, 255),
        2,
        cv::LINE_8);

    emit cropROIed(ImageUtils::matToQImage(cur_fg), ImageUtils::matToQImage(cropGoods), ImageUtils::matToQImage(copy_bg), ImageUtils::matToQImage(mask));

    QList<RecogResult> recogResult;
    if (cropGoods.empty()) {
        emit recogFinised(recogResult);

        emit error("未能获取商品，请确定已放置");
        return;
    }

    const auto text = detectText(cropGoods);

    std::vector<float> vec1 = m_featureEngine->extractFeatures(cropGoods, mask, text);

    std::vector<SearchResult> results = m_productDatabase->search(vec1);

    if (results.empty()) {
        emit error("未识别到商品");
    } else {
        emit error(QString("识别商品《%1》，相似度得分:%2").arg(QString::fromStdString(results.at(0).name)).arg(results.at(0).similarity));
    }

    qDebug() << "CropROI:" << t.elapsed();

    for (const auto& d : results) {
        auto val = std::find_if(recogResult.cbegin(), recogResult.cend(), [&d](const RecogResult& pre) {
            return pre.id == d.id;
        });

        if (val == recogResult.cend()) {
            recogResult.append({ d.tempId, d.id, d.name, d.barcode, d.similarity, d.recordIndex });
        } else {
            recogResult.append({ d.tempId, d.id, d.name, d.barcode, d.similarity, d.recordIndex, true });
        }
    }

    emit recogFinised(recogResult);
}

QString WorkThread::modelDir()
{
#ifdef Q_OS_ANDROID
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation).append("/model");
#elif defined Q_OS_WIN
    QString tempDir = QCoreApplication::applicationDirPath().append("/model");
#endif
    return tempDir;
}

QString WorkThread::ncnnModelDir()
{
#ifdef Q_OS_ANDROID
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation).append("/ncnn_models");
#elif defined Q_OS_WIN
    QString tempDir = QCoreApplication::applicationDirPath().append("/ncnn_models");
#endif
    return tempDir;
}

void WorkThread::initDir()
{
    QDir dir;
    dir.mkpath(modelDir());
    dir.mkpath(ncnnModelDir());
}

cv::Mat WorkThread::cropVideoFrame(const QImage& frame)
{
    cv::Rect roi = m_cropRect;

    if (frame.isNull() || roi.width <= 0 || roi.height <= 0)
        return cv::Mat();

    cv::Mat croppedMat;

    int height = frame.height();
    int width = frame.width();

    auto real = frame;

    if (m_mirrored == QCamera::FrontFace) {
        real = real.mirrored();
    }

    cv::Mat fullFrame = ImageUtils::qImageToMat(real);

    cv::Mat rotationFrame;

    auto rotaition = ((m_rotationAngle % 360) + 360) % 360;

    switch (rotaition) {
    case 90:
        cv::rotate(fullFrame, rotationFrame, cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE);
        std::swap(width, height);
        break;
    case 180:
        cv::rotate(fullFrame, rotationFrame, cv::RotateFlags::ROTATE_180);
        break;
    case 270:
        cv::rotate(fullFrame, rotationFrame, cv::RotateFlags::ROTATE_90_CLOCKWISE);
        std::swap(width, height);
        break;
    default:
        rotationFrame = fullFrame;
        break;
    }

    croppedMat = rotationFrame(roi & cv::Rect(0, 0, width, height)).clone();

    return croppedMat;
}

void WorkThread::loadBG()
{
    m_bg = cv::imread(Camera::backgroundImgDir().toStdString() + "/background.jpg");
    if (m_bg.empty()) {
        emit error("背景加载失败");
    }
}

void WorkThread::initOcrModel()
{
    m_ocr.reset(new OcrLite());

    std::string detPath = ensureModelFile(":/models/models/ch_PP-OCRv3_det_infer_fp16", "ocr_det").toStdString();
    std::string clsPath = ensureModelFile(":/models/models/ch_ppocr_mobile_v2.0_cls_infer_fp16", "ocr_cls").toStdString();
    std::string recPath = ensureModelFile(":/models/models/ch_PP-OCRv3_rec_infer_fp16", "ocr_rec").toStdString();
    std::string keysPath = ensureModelFile(":/models/models/ocr_keys_v1.txt", "ocr_keys.txt").toStdString();

    m_ocr->setNumThread(4);
    m_ocr->initLogger(true, false, false);
    m_ocr->initModels(detPath, clsPath, recPath, keysPath);
}

QString WorkThread::ensureModelFile(const QString& qrcPath, const QString& fileName)
{
    const auto targetPath = QDir(ncnnModelDir()).filePath(fileName);

    if (qrcPath.endsWith("txt")) {
        if (!QFile::exists(targetPath)) {
            QFile::copy(qrcPath, targetPath);
            QFile::setPermissions(targetPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }

    } else {
        const QString param = ".param";

        QString paramTargetPath = targetPath + param;
        if (!QFile::exists(paramTargetPath)) {
            bool copy = QFile::copy(qrcPath + param, paramTargetPath);

            qDebug() << "copy:" << copy << "qrcPath + param:" << qrcPath + param << "paramTargetPath:" << paramTargetPath;

            QFile::setPermissions(paramTargetPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }

        const QString bin = ".bin";

        QString binTargetPath = targetPath + bin;
        if (!QFile::exists(binTargetPath)) {
            QFile::copy(qrcPath + bin, binTargetPath);
            QFile::setPermissions(binTargetPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }
    }

    return targetPath;
}

std::string WorkThread::detectText(const cv::Mat& cur_fg)
{
    // int padding = 5; // 原 50 -> 缩减边缘 Padding 填充，减少无用像素计算
    // int maxSideLen = 320; // 原 1024 -> 降低检测输入分辨率（核心！计算量减少近 60%）
    // float boxScoreThresh = 0.80f; // 原 0.5f -> 提高框过滤阈值，少跑无用杂质框的 Rec 推理
    // float boxThresh = 0.8f;
    // float unClipRatio = 1.5f; // 原 1.6f -> 稍缩减文字框扩张比例
    // bool doAngle = true; // 原 true -> 禁用文字方向分类（核心！直接跳过所有文本框的 Cls 推理）
    // bool mostAngle = false; // 原 true -> 关闭全局角度校正

    // cv::Mat textMat;
    // cv::cvtColor(cur_fg, textMat, cv::COLOR_BGR2RGB);

    // OcrResult result = m_ocr->detect(textMat, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);

    // qInfo() << "txt:" << QString::fromStdString(result.strRes);

    // auto clear = m_textCleaner.clean(result.strRes);

    // auto txt = QString::fromStdString(clear);

    // qInfo() << "clean txt:" << txt;

    return {};
}

void WorkThread::initYoloModel()
{
    m_yoloDetector.reset(new Yolo11Ncnn());

    std::string path = ensureModelFile(":/models/models/yolo26n.ncnn", "yolo26n.ncnn").toStdString();

    m_yoloDetector->load_model(path + ".param", path + ".bin", 320, false);
}

void WorkThread::handleFrame()
{
    QImage snapshot;
    {
        QMutexLocker l(&m_lock);
        snapshot = m_frame_lst[m_curIndex];
    }

    auto data = cropVideoFrame(snapshot);

    if (!data.empty()) {

        cv::imwrite("./data.jpg", data);

        if (isStudied.loadRelaxed()) {
            isStudied.storeRelaxed(0);

            studied(data);
        } else if (isRecognition.loadRelaxed()) {
            isRecognition.storeRelaxed(0);

            recognition(data);
        } else {

            cv::Mat frame = data;

            if (data.channels() == 4)
                cv::cvtColor(data, frame, cv::COLOR_BGRA2BGR);
            else if (data.channels() == 1)
                cv::cvtColor(data, frame, cv::COLOR_GRAY2BGR);

            // std::vector<Object> objects;

            // m_yoloDetector->detect(frame, objects, 0.5f, 0.45f);

            std::vector<Detection> objects = m_yoloDetectorHand->detect(frame);

            for (const auto& det : objects) {
                if (det.label == 0 && det.prob > 0.01) {
                    qDebug() << "detect hand:"
                             << "true"
                             << " label : " << det.label << " prob:" << det.prob << "size:" << objects.size();
                }
            }

            static const QStringList COCO_CLASSES = {
                QStringLiteral("人"), QStringLiteral("自行车"), QStringLiteral("汽车"), QStringLiteral("摩托车"), QStringLiteral("飞机"), QStringLiteral("公交车"), QStringLiteral("火车"), QStringLiteral("卡车"), QStringLiteral("船"), QStringLiteral("红绿灯"),
                QStringLiteral("消防栓"), QStringLiteral("停止标志"), QStringLiteral("停车计时器"), QStringLiteral("长椅"), QStringLiteral("鸟"), QStringLiteral("猫"), QStringLiteral("狗"), QStringLiteral("马"), QStringLiteral("羊"), QStringLiteral("牛"),
                QStringLiteral("大象"), QStringLiteral("熊"), QStringLiteral("斑马"), QStringLiteral("长颈鹿"), QStringLiteral("背包"), QStringLiteral("雨伞"), QStringLiteral("手提包"), QStringLiteral("领带"), QStringLiteral("皮箱"), QStringLiteral("飞盘"),
                QStringLiteral("滑雪板"), QStringLiteral("单板滑雪"), QStringLiteral("运动球"), QStringLiteral("风筝"), QStringLiteral("棒球棒"), QStringLiteral("棒球手套"), QStringLiteral("滑板"), QStringLiteral("冲浪板"), QStringLiteral("网球拍"), QStringLiteral("瓶子"),
                QStringLiteral("高脚杯"), QStringLiteral("杯子"), QStringLiteral("叉子"), QStringLiteral("刀"), QStringLiteral("勺子"), QStringLiteral("碗"), QStringLiteral("香蕉"), QStringLiteral("苹果"), QStringLiteral("三明治"), QStringLiteral("橙子"),
                QStringLiteral("西兰花"), QStringLiteral("胡萝卜"), QStringLiteral("热狗"), QStringLiteral("披萨"), QStringLiteral("甜甜圈"), QStringLiteral("蛋糕"), QStringLiteral("椅子"), QStringLiteral("沙发"), QStringLiteral("盆栽"), QStringLiteral("床"),
                QStringLiteral("餐桌"), QStringLiteral("马桶"), QStringLiteral("电视"), QStringLiteral("笔记本"), QStringLiteral("鼠标"), QStringLiteral("遥控器"), QStringLiteral("键盘"), QStringLiteral("手机"), QStringLiteral("微波炉"), QStringLiteral("烤箱"),
                QStringLiteral("面包机"), QStringLiteral("洗碗机"), QStringLiteral("冰箱"), QStringLiteral("书"), QStringLiteral("时钟"), QStringLiteral("花瓶"), QStringLiteral("剪刀"), QStringLiteral("泰迪熊"), QStringLiteral("吹风机"), QStringLiteral("牙刷")
            };

            QVector<DetectObject> detectObject;
            detectObject.reserve(objects.size());

            for (const auto& obj : objects) {

                auto class_name = (obj.label >= 0 && obj.label < (int)COCO_CLASSES.size()) ? COCO_CLASSES[obj.label] : ("Class " + QString::number(obj.label));

                auto label_str = class_name;

                detectObject.append({ { obj.rect.x + m_cropRect.x, obj.rect.y + m_cropRect.y, obj.rect.width, obj.rect.height }, label_str, obj.prob });
            }

            emit detected(detectObject);
        }
    }
    m_hanlde_frame = false;
}

void WorkThread::initU2netpModel()
{

    BboxU2netp::Config cfg;
    cfg.useVulkan = false;

#if defined(__ANDROID__)
    cfg.numThreads = 2;
    cfg.useFP16 = true;
#else
    cfg.numThreads = 6;
    cfg.useFP16 = true;
#endif

    m_u2netp.reset(new BboxU2netp);

    m_u2netp->setConfig(cfg);

    std::string path = ensureModelFile(":/models/models/u2netp", "u2netp").toStdString();
    m_u2netp->load_model(path + ".param", path + ".bin");
}

void WorkThread::initISNetModel()
{
    m_ISNetSegmenter.reset(new ISNetSegmenter);
    m_ISNetSegmenter->load_model("ncnn_models/isnet_lite.ncnn.param", "ncnn_models/isnet_lite.ncnn.bin");
}

void WorkThread::initLiteSaliencyNetModel()
{
    LiteSaliencyNet::Config cfg;
    cfg.useVulkan = false;

#if defined(__ANDROID__)
    cfg.numThreads = 2;
    cfg.useFP16 = true;
#else
    cfg.numThreads = 6;
    cfg.useFP16 = true;
#endif

    m_LiteSaliencyNet.reset(new LiteSaliencyNet);

    m_LiteSaliencyNet->setConfig(cfg);

    std::string path = ensureModelFile(":/models/models/lite_320", "lite_320").toStdString();
    m_LiteSaliencyNet->load_model(path + ".param", path + ".bin");
}

void WorkThread::initYoloHandModel()
{
    m_yoloDetectorHand.reset(new HandDetector);
    std::string path = ensureModelFile(":/models/models/yolo26n_hand_320", "yolo26n_hand_320").toStdString();
    m_yoloDetectorHand->load_model(path + ".param", path + ".bin");
}

void WorkThread::studied(const cv::Mat& cur_fg)
{
    if (m_studiedRecord.name.empty()) {
        emit error("商品名称不能为空");
        return;
    }

    auto [cropGoods, mask] = m_LiteSaliencyNet->liteSaliencyCropROI(cur_fg);
    if (cropGoods.empty()) {
        emit error("学习失败，未能获取商品，请确定已放置");
        return;
    }

    const auto text = detectText(cropGoods);

    m_studiedRecord.feature = m_featureEngine->extractFeatures(cropGoods, mask, text);

    if (m_studiedRecord.id == 0) {
        m_studiedRecord.id = (quint64)QDateTime::currentSecsSinceEpoch();
    }

    bool code = m_productDatabase->addProduct(m_studiedRecord);
    if (code) {
        emit error("学习成功");
    } else {
        emit error("学习失败");
    }

    m_studiedRecord = { 0 };
}

#include <filesystem>
void WorkThread::splitData()
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
