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

        m_productDatabase.loadFromFile("./model/data.bat");

        loadBG();
    });

    m_featureWeights.color = 0.6f; // 调大可增加对颜色的敏感度
    m_featureWeights.texture = 0.8f; // 调大可增加对表面花纹的敏感度
    m_featureWeights.shape = 1.0f;
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

    cv::Mat cropBg = m_bg(m_cropRect);

    // 2. 自动抠图获得纯净 ROI
    auto [cropGoods, mask] = ProductFeatureEngine::cropROI(cropBg, cur_fg);

    cv::Mat copy_bg = m_bg.clone();
    cv::rectangle(copy_bg,
        m_cropRect & cv::Rect(0, 0, m_bg.cols, m_bg.rows),
        cv::Scalar(0, 0, 255),
        2,
        cv::LINE_8);

    emit cropROIed(ImageUtils::matToQImage(cropGoods), ImageUtils::matToQImage(copy_bg), ImageUtils::matToQImage(mask));

    if (cropGoods.empty()) {
        emit recogFinised({});

        emit error("未能获取商品，请确定已放置");
        return;
    }

    // 3. 提取 391 维加权归一化向量
    std::vector<float> vec1 = ProductFeatureEngine::extract(cropGoods, mask, m_featureWeights);

    std::vector<SearchResult> results = m_productDatabase.search(vec1);

    if (results.empty()) {
        emit error("未识别到商品");
    } else {
        emit error(QString("识别商品《%1》，相似度得分:%2").arg(QString::fromStdString(results.at(0).name)).arg(results.at(0).similarity));
    }

    emit recogFinised(QList<SearchResult>(results.begin(), results.end()));
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

    m_studiedRecord.feature = ProductFeatureEngine::extract(cropGoods, mask, m_featureWeights);

    auto existFeature = m_productDatabase.searchVec(m_studiedRecord.id);

    ProductFeatureEngine::reinforceFeatureVector(m_studiedRecord.feature, existFeature);

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
