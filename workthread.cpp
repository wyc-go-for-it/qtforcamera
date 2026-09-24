#include "workthread.h"
#include "camera.h"
#include "imageUtils.h"
#include "opencv2/imgcodecs.hpp"
#include "productfeatureengine.h"
#include "qdebug.h"

#include <opencv2/core/mat.hpp>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>

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

    connect(&m_thread, &QThread::started, this, [this]() {
        initDir();

        initOcrModel();

        m_productDatabase.loadModel((modelDir() + "/data.bat").toStdString());

        loadBG();
    });
}

WorkThread::~WorkThread()
{
    m_productDatabase.saveModel((modelDir() + "/data.bat").toStdString());

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

void WorkThread::onVideoFrameChanged(const QVideoFrame& frame)
{
    if (!frame.isValid())
        return;

    if (isStudied.loadRelaxed()) {
        isStudied.storeRelaxed(0);
        QVideoFrame tmp(frame);
        auto data = cropVideoFrame(tmp);

        studied(data);
    } else if (isRecognition.loadRelaxed()) {
        isRecognition.storeRelaxed(0);

        QVideoFrame tmp(frame);
        auto data = cropVideoFrame(tmp);

        recognition(data);
    }
}

void WorkThread::onRecognition(bool recog)
{
    isRecognition.storeRelaxed(recog);
}

void WorkThread::onUpdateVertex(const QVector<QPointF>& points)
{
    m_cropPoints.clear();
    m_cropPoints.append(points);

    std::vector<cv::Point> pts;
    pts.reserve(m_cropPoints.size());
    for (const QPointF& pt : qAsConst(m_cropPoints)) {
        pts.emplace_back(qRound(pt.x()), qRound(pt.y()));
    }

    m_cropRect = cv::boundingRect(pts);
}

void WorkThread::recognition(const cv::Mat& cur_fg)
{
    if (m_bg.empty()) {
        emit error("背景无效");
        return;
    }

    if (cur_fg.empty()) {
        emit error("识别图片转换错误");
        return;
    }

    cv::Mat cropBg = m_bg(m_cropRect & cv::Rect(0, 0, m_bg.cols, m_bg.rows));

    // 2. 自动抠图获得纯净 ROI
    auto [cropGoods, mask] = ProductFeatureEngine::cropROI(cropBg, cur_fg);

    cv::Mat copy_bg = m_bg.clone();
    cv::rectangle(copy_bg,
        m_cropRect & cv::Rect(0, 0, m_bg.cols, m_bg.rows),
        cv::Scalar(0, 0, 255),
        2,
        cv::LINE_8);

    emit cropROIed(ImageUtils::matToQImage(cropGoods), ImageUtils::matToQImage(copy_bg), ImageUtils::matToQImage(mask));

    QList<RecogResult> recogResult;
    if (cropGoods.empty()) {
        emit recogFinised(recogResult);

        emit error("未能获取商品，请确定已放置");
        return;
    }

    const auto text = detectText(cropGoods);

    std::vector<float> vec1 = ProductFeatureEngine::fuseFeatures(cropGoods, mask, text);

    std::vector<SearchResult> results = m_productDatabase.search(vec1);

    if (results.empty()) {
        emit error("未识别到商品");
    } else {
        emit error(QString("识别商品《%1》，相似度得分:%2").arg(QString::fromStdString(results.at(0).name)).arg(results.at(0).similarity));
    }

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

QString WorkThread::ocrModelDir()
{
#ifdef Q_OS_ANDROID
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation).append("/ocr_models");
#elif defined Q_OS_WIN
    QString tempDir = QCoreApplication::applicationDirPath().append("/ocr_models");
#endif
    return tempDir;
}

void WorkThread::initDir()
{
    QDir dir;
    dir.mkpath(modelDir());
    dir.mkpath(ocrModelDir());
}

cv::Mat WorkThread::cropVideoFrame(QVideoFrame& frame)
{
    cv::Rect roi = m_cropRect;

    if (!frame.isValid() || roi.width <= 0 || roi.height <= 0)
        return cv::Mat();

    cv::Mat croppedMat;

    if (frame.map(QAbstractVideoBuffer::ReadOnly)) {

        int height = frame.height();
        int width = frame.width();

        cv::Mat fullFrame = ImageUtils::qImageToMat(frame.image().mirrored());

        croppedMat = fullFrame(roi & cv::Rect(0, 0, width, height)).clone();

        frame.unmap();
    }

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
    std::string detPath = ensureModelFile(":/models/models/ch_PP-OCRv3_det_infer_fp16", "det").toStdString();
    std::string clsPath = ensureModelFile(":/models/models/ch_ppocr_mobile_v2.0_cls_infer_fp16", "cls").toStdString();
    std::string recPath = ensureModelFile(":/models/models/ch_PP-OCRv3_rec_infer_fp16", "rec").toStdString();
    std::string keysPath = ensureModelFile(":/models/models/ocr_keys_v1.txt", "keys.txt").toStdString();

    m_ocr.initLogger(true, false, false);
    m_ocr.initModels(detPath, clsPath, recPath, keysPath);
}

QString WorkThread::ensureModelFile(const QString& qrcPath, const QString& fileName)
{
    const auto targetPath = QDir(ocrModelDir()).filePath(fileName);

    if (qrcPath.endsWith("txt")) {
        if (!QFile::exists(targetPath)) {
            QFile::copy(qrcPath, targetPath);
            QFile::setPermissions(targetPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }

    } else {
        const QString param = ".param";

        QString paramTargetPath = targetPath + param;
        if (!QFile::exists(paramTargetPath)) {
            QFile::copy(qrcPath + param, paramTargetPath);
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
    int padding = 5; // 原 50 -> 缩减边缘 Padding 填充，减少无用像素计算
    int maxSideLen = std::max(cur_fg.rows, cur_fg.cols); // 原 1024 -> 降低检测输入分辨率（核心！计算量减少近 60%）
    float boxScoreThresh = 0.80f; // 原 0.5f -> 提高框过滤阈值，少跑无用杂质框的 Rec 推理
    float boxThresh = 0.8f;
    float unClipRatio = 1.5f; // 原 1.6f -> 稍缩减文字框扩张比例
    bool doAngle = true; // 原 true -> 禁用文字方向分类（核心！直接跳过所有文本框的 Cls 推理）
    bool mostAngle = false; // 原 true -> 关闭全局角度校正

    cv::Mat textMat;
    cv::cvtColor(cur_fg, textMat, cv::COLOR_BGR2RGB);

    OcrResult result = m_ocr.detect(textMat, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);

    qInfo() << "txt:" << QString::fromStdString(result.strRes);

    auto clear = m_textCleaner.clean(result.strRes);

    auto txt = QString::fromStdString(clear);

    qInfo() << "clean txt:" << QString::fromStdString(clear);

    return clear;
}

void WorkThread::studied(const cv::Mat& cur_fg)
{
    if (m_studiedRecord.name.empty()) {
        emit error("商品名称不能为空");
        return;
    }

    auto [cropGoods, mask] = ProductFeatureEngine::cropROI(m_bg(m_cropRect), cur_fg);
    if (cropGoods.empty()) {
        emit error("学习失败，未能获取商品，请确定已放置");
        return;
    }

    const auto text = detectText(cropGoods);

    m_studiedRecord.feature = ProductFeatureEngine::fuseFeatures(cropGoods, mask, text);

    if (m_studiedRecord.id == 0) {
        m_studiedRecord.id = (quint64)QDateTime::currentSecsSinceEpoch();
    }

    bool code = m_productDatabase.addProduct(m_studiedRecord);
    if (code) {
        emit error("学习成功");
    } else {
        emit error("学习失败");
    }

    m_studiedRecord = { 0 };
}
