#ifndef WORKTHREAD_H
#define WORKTHREAD_H

#include "productdatabase.h"
#include "productfeatureengine.h"
#include "qthread.h"

#include <QVideoFrame>

class WorkThread : public QObject {
    Q_OBJECT

public:
    explicit WorkThread(QObject* parent = nullptr);
    ~WorkThread();

    void init();
    void studying(const QString& barcode, const QString& name);

public slots:
    void onVideoFrameChanged(const QVideoFrame& frame);
    void onRecognition(bool recog);
    void onUpdateVertex(const QVector<QPointF>& points);

signals:
    void cropROIed(const QImage& diff, const QImage& undiff, const QImage& binary_diff);
    void error(const QString& error);

private:
    void studied(const cv::Mat& cur_fg);
    void recognition(const cv::Mat& cur_fg);
    QString modelDir();
    void initDir();
    cv::Mat cropVideoFrame(QVideoFrame& frame);

private:
    QThread m_thread;
    QAtomicInt isRecognition, isStudied;
    ProductDatabase m_productDatabase;
    cv::Mat m_bg;
    FeatureWeights m_featureWeights;
    QString m_barcode;
    QString m_name;
    QVector<QPointF> m_cropPoints; // 0:左上, 1:右上, 2:右下, 3:左下
    cv::Rect m_cropRect;
};

#endif // WORKTHREAD_H
