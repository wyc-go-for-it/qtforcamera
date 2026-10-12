#ifndef WORKTHREAD_H
#define WORKTHREAD_H

#include "OcrLite.h"
#include "productdatabase.h"
#include "producttextcleaner.h"
#include "qthread.h"
#include "yoloncnn.h"

#include <QMutex>
#include <QVideoFrame>

struct RecogResult {
    uint64_t tempId;
    uint64_t id;
    std::string name;
    std::string barcode;
    float similarity;
    size_t recordIndex;

    bool show = false;
};
Q_DECLARE_METATYPE(RecogResult);

struct DetectObject {
    QRectF rect; // 目标框 [x, y, width, height]
    QString name; // 类别
    float prob; // 置信度
};
Q_DECLARE_METATYPE(DetectObject);

class WorkThread : public QObject {
    Q_OBJECT

public:
    explicit WorkThread(QObject* parent = nullptr);
    ~WorkThread();

    void init();
    void studying(const ProductRecord& record);

public slots:
    void onVideoFrameChanged(const QVideoFrame& frame);
    void onRecognition(bool recog);
    void onUpdateVertex(const QVector<QPointF>& points);

signals:
    void cropROIed(const QImage& frame, const QImage& diff, const QImage& undiff, const QImage& binary_diff);
    void error(const QString& error);
    void recogFinised(const QList<RecogResult>& data);

    void detected(const QVector<DetectObject>& objects);

private:
    void studied(const cv::Mat& cur_fg);
    void recognition(const cv::Mat& cur_fg);
    QString modelDir();
    QString ncnnModelDir();
    void initDir();
    cv::Mat cropVideoFrame(QVideoFrame& frame);
    void loadBG();
    void initOcrModel();

    QString ensureModelFile(const QString& qrcPath, const QString& fileName);

    std::string detectText(const cv::Mat& cur_fg);

    void initYoloModel();

    Q_INVOKABLE void handleFrame();

private:
    QThread m_thread;
    QAtomicInt isRecognition, isStudied;
    cv::Mat m_bg;
    QList<QVideoFrame> m_frame_lst;
    QMutex m_lock;
    bool m_hanlde_frame = false;

    ProductDatabase m_productDatabase;
    ProductRecord m_studiedRecord;
    OcrLite m_ocr;
    ProductTextCleaner m_textCleaner;

    QVector<QPointF> m_cropPoints; // 0:左上, 1:右上, 2:右下, 3:左下
    cv::Rect m_cropRect;

    Yolo11Ncnn m_yoloDetector;
};

#endif // WORKTHREAD_H
