#include "faceworkthread.h"
#include "imageUtils.h"
#include "opencv2/imgproc.hpp"
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>
#include <face/FaceEngine.h>
#include <face/ImageAugmentor.h>
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <random>

FaceWorkThread::FaceWorkThread(QObject* parent)
    : QObject(parent)
{
    moveToThread(&m_thread);

    m_FaceDatabase.reset(new ProductDatabase(FaceEngine::featureDim));

    connect(&m_thread, &QThread::started, this, [this]() {
        initDir();
        initFaceModel();

        m_FaceDatabase->loadModel((WorkThread::modelDir() + "/face_data.bat").toStdString());
        emit info(QString("加载<%1>条数据").arg(m_FaceDatabase->size()));
    });
}

FaceWorkThread::~FaceWorkThread()
{
    m_FaceDatabase->saveModel((WorkThread::modelDir() + "/face_data.bat").toStdString());

    m_thread.quit();
    m_thread.wait(3000);
}

void FaceWorkThread::init()
{
    m_thread.start();
}

void FaceWorkThread::onVideoFrameChanged(const QVideoFrame& frame)
{
    if (!frame.isValid())
        return;

    QMutexLocker l(&m_lock);
    if (m_frame_lst.size() < 3) {
        m_frame_lst.append(frame);
    } else {
        m_frame_lst[0] = m_frame_lst[1];
        m_frame_lst[1] = m_frame_lst[2];
        m_frame_lst[2] = frame;
    }

    if (!m_hanlde_frame) {
        QMetaObject::invokeMethod(this, "handleFrame", Qt::QueuedConnection);
    }
}

void FaceWorkThread::onRegisteredFace(bool recog, const QString& name)
{
    m_name = name;
    isRegistered.storeRelaxed(recog);
}

void FaceWorkThread::onQueriedFace(bool recog, const QString& name)
{
    m_name = name;
    isQueried.storeRelaxed(recog);
}

void FaceWorkThread::onUpdateVertex(QVector<QPointF> points, QCamera::Position pos, qint32 rotationAngle)
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

void FaceWorkThread::onExtractFeature()
{
    QMetaObject::invokeMethod(this, "extractFeature", Qt::QueuedConnection, Q_ARG(QFileInfo, QFileInfo()));
}

void FaceWorkThread::onRandomTest()
{
    QMetaObject::invokeMethod(this, "testFace", Qt::QueuedConnection);
}

void FaceWorkThread::testSimilarity(const cv::Mat& base_img, const cv::Mat& test_img, const QString& test_name)
{
    auto base_faceLst = m_faceDetector->detectFace(base_img);
    auto test_faceLst = m_faceDetector->detectFace(test_img);

    if (base_faceLst.empty() || test_faceLst.empty()) {
        emit info(test_name + "，未发现查询人脸信息");
        return;
    }

    m_faceDetector->extract_feature(base_faceLst.at(0));
    auto base_feat = base_faceLst.at(0).feature;

    m_faceDetector->extract_feature(test_faceLst.at(0));
    auto test_feat = test_faceLst.at(0).feature;

    emit captured(ImageUtils::matToQImage(base_img), ImageUtils::matToQImage(test_img), ImageUtils::matToQImage(base_faceLst.at(0).aligned));

    if (base_feat.empty() || test_feat.empty()) {
        emit info("[" + test_name + "] ❌ 无法提取特征 (可能人脸检测失败)");
        return;
    }

    float sim = FaceEngine::cosine_similarity(base_feat, test_feat);

    qInfo() << QString("[%1-15s] 相似度: %2").arg(test_name).arg(sim);

    emit info(QString("[%1-15s] 相似度: %2").arg(test_name).arg(sim));

    QThread::msleep(500);
}

void FaceWorkThread::testFace()
{

    namespace fs = std::filesystem;

    const auto src_dir_path = "./all_faces";

    qInfo() << "========== 开始鲁棒性测试 ==========";

    static bool is_init = false;
    static size_t index = 0;
    static std::vector<std::string> base_img_files, test_img_files;

    if (!is_init) {
        for (const auto& entry : fs::directory_iterator(src_dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    base_img_files.push_back(entry.path().string());
                    test_img_files.push_back(entry.path().string());
                }
            }
        }

        std::random_device rd;
        std::default_random_engine eng(rd());
        std::shuffle(base_img_files.begin(), base_img_files.end(), eng);
        std::shuffle(test_img_files.begin(), test_img_files.end(), eng);

        is_init = true;
    }

    if (index > base_img_files.size())
        return;

    const cv::Mat base_raw = cv::imread(base_img_files.at(index++));

    const cv::Mat test_raw = cv::imread(test_img_files.at(test_img_files.size() - index - 1));

    // 基准图与自己的测试 (分数应该接近 1.0)
    testSimilarity(base_raw, test_raw, "基准图 vs 基准图");

    // 2. 亮度测试
    testSimilarity(base_raw, ImageAugmentor::adjustBrightness(test_raw, 50), "亮度+50(偏亮)");
    testSimilarity(base_raw, ImageAugmentor::adjustBrightness(test_raw, -50), "亮度-50(偏暗)");
    testSimilarity(base_raw, ImageAugmentor::adjustBrightness(test_raw, -80), "亮度-80(极暗)");

    // 3. 对比度测试
    testSimilarity(base_raw, ImageAugmentor::adjustContrast(test_raw, 0.5f), "对比度0.5(灰)");
    testSimilarity(base_raw, ImageAugmentor::adjustContrast(test_raw, 1.5f), "对比度1.5(强)");

    // 4. Gamma 测试 (模拟不同相机的感光曲线)
    testSimilarity(base_raw, ImageAugmentor::adjustGamma(test_raw, 0.6), "Gamma0.6(提暗)");
    testSimilarity(base_raw, ImageAugmentor::adjustGamma(test_raw, 1.5), "Gamma1.5(压暗)");

    // 5. 颜色/色温测试
    testSimilarity(base_raw, ImageAugmentor::adjustHSV(test_raw, 15, 1.0f, 1.0f), "色相偏移(偏色)");
    testSimilarity(base_raw, ImageAugmentor::adjustHSV(test_raw, 0, 0.5f, 1.0f), "低饱和度(褪色)");

    // 6. 画质模糊/噪声测试
    testSimilarity(base_raw, ImageAugmentor::addGaussianNoise(test_raw, 0, 15), "高斯噪声(std=15)");
    testSimilarity(base_raw, ImageAugmentor::addBlur(test_raw, 5), "高斯模糊(5x5)");

    // 7. 几何变换测试 (★ 核心：考验对齐算法)
    testSimilarity(base_raw, ImageAugmentor::rotate(test_raw, 15.0f), "向右旋转15度");
    testSimilarity(base_raw, ImageAugmentor::rotate(test_raw, -15.0f), "向左旋转15度");
    testSimilarity(base_raw, ImageAugmentor::scale(test_raw, 0.4f), "缩放0.4(偏远)");

    // 8. 组合蹂躏测试 (模拟地狱环境)
    cv::Mat hell_img = ImageAugmentor::adjustBrightness(test_raw, -40);
    hell_img = ImageAugmentor::addBlur(hell_img, 3);
    hell_img = ImageAugmentor::addGaussianNoise(hell_img, 0, 10);
    testSimilarity(base_raw, hell_img, "暗光+模糊+噪声");

    qInfo() << "========== 测试结束 ==========";

    onRandomTest();
}

void FaceWorkThread::registerFace(const cv::Mat& cur_fg, const QString& name)
{
    if (name.isEmpty() && m_name.isEmpty()) {
        emit error("名字不能为空");
        return;
    }

    std::vector<FaceInfo> faces = m_faceDetector->detectFace(cur_fg);
    if (faces.size() != 1) {
        emit info("一次只能注册一个人");
    } else {
        ProductRecord faceRecord;
        faceRecord.id = (quint64)QDateTime::currentMSecsSinceEpoch();
        faceRecord.name = name.isEmpty() ? "名字_" + std::to_string(faceRecord.id) : name.toStdString();

        m_faceDetector->extract_feature(faces.at(0));

        faceRecord.feature = faces.at(0).feature;

        m_FaceDatabase->addProduct(faceRecord);

        QString filePath = faceDataDir() + "/" + QString::fromStdString(faceRecord.name) + ".jpg";

        QImage qimg(ImageUtils::matToQImage(cur_fg));
        qimg.save(filePath, "JPG");

        filePath = faceDataDir() + "/" + QString::fromStdString(faceRecord.name) + "_aligned.jpg";
        QImage aligned(ImageUtils::matToQImage(faces.at(0).aligned));
        aligned.save(filePath, "JPG");

        emit captured(qimg, qimg, aligned);
    }
    m_name.clear();
}

cv::Mat addGaussianNoise(const cv::Mat& src, double mean, double stddev)
{
    cv::Mat noise(src.size(), src.type());
    cv::randn(noise, mean, stddev);
    cv::Mat dst;
    cv::add(src, noise, dst);
    return dst;
}

void FaceWorkThread::queryFace(const cv::Mat& cur_fg)
{
    cv::Mat cur_data = cur_fg;
    if (!m_name.isEmpty()) {
        cur_data = cv::imread(m_name.toLocal8Bit().toStdString());
    }

    std::vector<FaceInfo> faces = m_faceDetector->detectFace(cur_data);
    if (faces.empty()) {
        emit captured(ImageUtils::matToQImage(cur_data), QImage(), QImage());
        emit info("未发现查询人脸信息");
    } else {
        m_faceDetector->extract_feature(faces.at(0));

        QElapsedTimer t;
        t.start();

        std::vector<SearchResult> results = m_FaceDatabase->search(faces.at(0).feature);

        if (results.empty()) {
            emit info("未识别到人脸信息");
        } else {
            emit info(QString("搜索耗时<%3ms>,识别人脸信息《%1》，相似度得分:%2").arg(QString::fromStdString(results.at(0).name)).arg(results.at(0).similarity).arg(t.elapsed()));

            auto name = QString("%1/%2.jpg").arg(faceDataDir(), QString::fromStdString(results.at(0).name));

            auto reg_data = cv::imread(name.toLocal8Bit().toStdString());

            name = QString("%1/%2_aligned.jpg").arg(faceDataDir(), QString::fromStdString(results.at(0).name));

            auto aligned = cv::imread(name.toLocal8Bit().toStdString());

            emit captured(ImageUtils::matToQImage(cur_data), ImageUtils::matToQImage(reg_data), ImageUtils::matToQImage(aligned));
        }
    }
}

QString FaceWorkThread::faceDataDir()
{
#ifdef Q_OS_ANDROID
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation).append("/face_data");
#elif defined Q_OS_WIN
    QString tempDir = QCoreApplication::applicationDirPath().append("/face_data");
#endif
    return tempDir;
}

void FaceWorkThread::initDir()
{
    QDir dir;
    dir.mkpath(faceDataDir());
}

cv::Mat FaceWorkThread::cropVideoFrame(QVideoFrame& frame)
{
    cv::Rect roi = m_cropRect;

    if (!frame.isValid() || roi.width <= 0 || roi.height <= 0)
        return cv::Mat();

    cv::Mat croppedMat;

    if (frame.map(QAbstractVideoBuffer::ReadOnly)) {

        int height = frame.height();
        int width = frame.width();

        auto real = frame.image();

        frame.unmap();

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
    }

    return croppedMat;
}

void FaceWorkThread::handleFrame()
{
    QMutexLocker l(&m_lock);
    if (m_frame_lst.isEmpty())
        return;

    m_hanlde_frame = true;

    auto data = cropVideoFrame(m_frame_lst.last());

    l.unlock();

    if (!data.empty()) {

        if (isRegistered.loadRelaxed()) {
            isRegistered.storeRelaxed(0);

            registerFace(data, m_name);
        } else if (isQueried.loadRelaxed()) {
            isQueried.storeRelaxed(0);

            queryFace(data);
        } else {
            std::vector<FaceInfo> faces = m_faceDetector->detectFace(data);

            QVector<DetectObject> detectObject;
            detectObject.reserve(faces.size());

            for (const auto& obj : faces) {

                if (!obj.is_real) {
                    emit error("请使用真人");
                }

                if (obj.isMouthOpen()) {
                    emit error("警告：张嘴了");
                }

                if (obj.allEyeClosed()) {
                    emit error("警告：闭眼了");
                } else if (obj.isLeftEyeClosed()) {
                    emit error("警告：左眼闭眼了");
                } else if (obj.isRightEyeClosed()) {
                    emit error("警告：右眼闭眼了");
                }

                if (obj.isHeadTilted()) {
                    emit error("警告：检测到歪头");
                }

                if (obj.isHeadTurnLeft()) {
                    emit error("警告：检测到大幅向左转头");
                }

                if (obj.isHeadTurnRight()) {
                    emit error("警告：检测到大幅向右转头");
                }

                if (obj.isHeadDown()) {
                    emit error("警告：检测到低头");
                }

                if (obj.isHeadUp()) {
                    emit error("警告：检测到仰头");
                }

                auto label_str = "人脸";

                QVector<QPointF> landmarks;
                QVector<QPointF> lm106;

                for (const auto& points : obj.landmarks) {
                    landmarks.append({ points.x + m_cropRect.x, points.y + m_cropRect.y });
                }

                for (const auto& points : obj.lm106) {
                    lm106.append({ points.x + m_cropRect.x, points.y + m_cropRect.y });
                }

                // float scale = 5.0f;
                // cv::Mat big_img;
                // cv::resize(data, big_img, cv::Size(), scale, scale, cv::INTER_CUBIC);

                // for (int i = 0; i < 106; ++i) {
                //     // 放大坐标
                //     int x = (int)(obj.lm106[i].x * scale);
                //     int y = (int)(obj.lm106[i].y * scale);

                //     // 2. 画红点
                //     cv::circle(big_img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);

                //     // 3. 画文字（使用黄色，带一点错位，防止完全覆盖点）
                //     std::string text = std::to_string(i);
                //     cv::putText(big_img, text, cv::Point(x + 3, y - 3),
                //         cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                // }

                // // 4. 保存图片
                // cv::imwrite("./draw_192.jpg", big_img);

                detectObject.append({ { obj.rect.x + m_cropRect.x, obj.rect.y + m_cropRect.y, obj.rect.width, obj.rect.height }, label_str, obj.score, landmarks, lm106 });
            }

            emit detected(detectObject);
        }
    }
    m_hanlde_frame = false;
}

void FaceWorkThread::extractFeature(const QFileInfo& fileInfo)
{

    QString qImgDir = "./all_faces";

    // 2. 扩展名过滤
    QStringList exts = { "jpg", "jpeg", "png", "bmp" };
    int processed = 0, skipped = 0;

    QString last_path;

    // 3. 遍历目录
    QDir dir(qImgDir);
    // 按照文件名排序，保证每次处理的顺序一致（方便调试）
    dir.setSorting(QDir::Name);
    QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);

    if (fileList.isEmpty()) {
        emit info("⚠️ 警告：目录为空或不存在:" + qImgDir);
        return;
    }

    emit info("开始处理目录:" + qImgDir + "，共发现文件:" + QString::number(fileList.size()));

    for (const QFileInfo& fileInfo : fileList) {
        QString path = fileInfo.absoluteFilePath();
        QString ext = fileInfo.suffix().toLower();

        if (!exts.contains(ext))
            continue;

        // 4. 读取图片
        cv::Mat img = cv::imread(path.toStdString());
        if (img.empty()) {
            emit info("读取失败:" + path);
            skipped++;
            continue;
        }

        emit info("正在处理:" + path);

        registerFace(img, fileInfo.completeBaseName());

        // ... 你的检测逻辑 ...

        // 示例：进度打印（每 50 张打印一次）
        if (processed % 50 == 0 && processed > 0) {
            emit info("已处理:" + QString::number(processed) + "张 | 跳过:" + QString::number(skipped));
        }
    }

    emit info("处理完成！成功:" + QString::number(processed) + "张 | 跳过:" + QString::number(skipped));
}

void FaceWorkThread::initFaceModel()
{
    m_faceDetector.reset(new FaceEngine);

    std::string det_path = WorkThread::ensureModelFile(":/models/models/det_2.5g.ncnn", "det_2.5g.ncnn").toStdString();

    std::string rec_path = WorkThread::ensureModelFile(":/models/models/w600k_r50.ncnn", "w600k_r50.ncnn").toStdString();

    std::string lm_path = WorkThread::ensureModelFile(":/models/models/2d106det.ncnn", "2d106det.ncnn").toStdString();

    std::string silentFace_path = WorkThread::ensureModelFile(":/models/models/MiniFASNetV2.ncnn", "MiniFASNetV2.ncnn").toStdString();

    m_faceDetector->loadDetModel(det_path + ".param", det_path + ".bin");
    m_faceDetector->loadRecModel(rec_path + ".param", rec_path + ".bin");
    m_faceDetector->loadLmModel(lm_path + ".param", lm_path + ".bin");
    m_faceDetector->loadSilentFaceAntiSpoofingModel(silentFace_path + ".param", silentFace_path + ".bin");
}
