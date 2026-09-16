#include "workthread.h"
#include "ImageUtils.h"
#include "camera.h"
#include "opencv2/imgcodecs.hpp"
#include "productfeatureengine.h"
#include "qdebug.h"

#include <opencv2/core/mat.hpp>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>

WorkThread::WorkThread(QObject* parent)
    : QObject(parent)
{
    moveToThread(&m_thread);

    connect(&m_thread, &QThread::started, this, [this]() {
        initDir();

        m_productDatabase.loadFromFile("./model/data.bat");
        m_bg = cv::imread(Camera::backgroundImgDir().toStdString() + "/background.jpg");
        if (m_bg.empty()) {
            emit error("背景加载失败");
        }
    });

    m_featureWeights.color = 1.0f; // 调大可增加对颜色的敏感度
    m_featureWeights.texture = 0.8f; // 调大可增加对表面花纹的敏感度
    m_featureWeights.shape = 0.5f;
}

WorkThread::~WorkThread()
{
    m_productDatabase.saveToFile("./model/data.bat");

    m_thread.quit();
    m_thread.wait(3000);
}

void WorkThread::init()
{
    m_thread.start();
}

void WorkThread::studying(const QString& barcode, const QString& name)
{
    m_barcode = barcode;
    m_name = name;
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
}

void WorkThread::recognition(const cv::Mat& cur_fg)
{

    if (cur_fg.empty()) {
        emit error("识别图片转换错误");
        return;
    }

    // 2. 自动抠图获得纯净 ROI
    auto [roi1, mask] = ProductFeatureEngine::cropROI(m_bg, cur_fg);

    emit cropROIed(ImageUtils::matToQImage(roi1), ImageUtils::matToQImage(m_bg), ImageUtils::matToQImage(mask));

    if (roi1.empty()) {
        emit error("未能获取商品，请确定已放置");
        return;
    }

    // 3. 提取 391 维加权归一化向量
    std::vector<float> vec1 = ProductFeatureEngine::extract(roi1, mask, m_featureWeights);

    std::vector<SearchResult> results = m_productDatabase.search(vec1);

    if (results.empty()) {
        emit error("未识别到商品");
    } else {

        emit error(QString("识别商品《%1》，相似度得分:%2").arg(QString::fromStdString(results.at(0).name)).arg(results.at(0).similarity));

        for (const auto& r : results) {

            qDebug() << "向量维度: " << vec1.size();
            qDebug() << "当前商品与商品<" << QString(r.name.c_str()) << ">"
                     << "的相似度得分: " << r.similarity;

            if (r.similarity > 0.85f) {
                qDebug() << "匹配结果: 同一款商品";
            } else {
                std::cout << "匹配结果: 不同商品";
            }
        }
    }
}

QString WorkThread::modelDir()
{
    return QCoreApplication::applicationDirPath() + "/model";
}

void WorkThread::initDir()
{
    QDir dir;
    dir.mkpath(modelDir());
}

cv::Mat WorkThread::cropVideoFrame(QVideoFrame& frame)
{
    std::vector<cv::Point> pts;
    pts.reserve(m_cropPoints.size());
    for (const QPointF& pt : qAsConst(m_cropPoints)) {
        pts.emplace_back(qRound(pt.x()), qRound(pt.y()));
    }

    cv::Rect roi = cv::boundingRect(pts);

    if (!frame.isValid() || roi.width <= 0 || roi.height <= 0)
        return cv::Mat();

    cv::Mat croppedMat;

    if (frame.map(QAbstractVideoBuffer::ReadOnly)) {

        int height = frame.height();
        int width = frame.width();
        int bytesPerLine = frame.bytesPerLine();

        cv::Mat fullFrame = ImageUtils::qImageToMat(frame.image().mirrored());

        cv::Rect safeRoi = roi & cv::Rect(0, 0, width, height);

        std::cout << "匹配结果: 不同商品" << safeRoi << std::endl;

        croppedMat = fullFrame(safeRoi).clone();

        frame.unmap();
    }

    return croppedMat;
}

void WorkThread::studied(const cv::Mat& cur_fg)
{
    auto [roi1, mask] = ProductFeatureEngine::cropROI(m_bg, cur_fg);
    if (roi1.empty()) {
        emit error("学习失败，未能获取商品，请确定已放置");
        return;
    }

    std::vector<float> feature = ProductFeatureEngine::extract(roi1, mask, m_featureWeights);

    m_productDatabase.addProduct({ (quint64)QDateTime::currentSecsSinceEpoch(), m_name.toStdString(), m_barcode.toStdString(), feature });
}
