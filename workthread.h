#ifndef WORKTHREAD_H
#define WORKTHREAD_H

#include "OcrLite.h"
#include "productdatabase.h"
#include "producttextcleaner.h"
#include "qthread.h"

#include <QVideoFrame>

struct RecogResult {
    uint64_t tempId;
    uint64_t id;
    std::string name;
    std::string barcode;
    float similarity;
    uint32_t recordIndex;

    bool show = false;
};
Q_DECLARE_METATYPE(RecogResult);

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
    void cropROIed(const QImage& diff, const QImage& undiff, const QImage& binary_diff);
    void error(const QString& error);
    void recogFinised(const QList<RecogResult>& data);

private:
    void studied(const cv::Mat& cur_fg);
    void recognition(const cv::Mat& cur_fg);
    QString modelDir();
    QString ocrModelDir();
    void initDir();
    cv::Mat cropVideoFrame(QVideoFrame& frame);
    void loadBG();
    void initOcrModel();

    QString ensureModelFile(const QString& qrcPath, const QString& fileName);

    std::string detectText(const cv::Mat& cur_fg);

private:
    QThread m_thread;
    QAtomicInt isRecognition, isStudied;
    cv::Mat m_bg;

    ProductDatabase m_productDatabase;
    ProductRecord m_studiedRecord;
    OcrLite m_ocr;
    ProductTextCleaner m_textCleaner;

    QVector<QPointF> m_cropPoints; // 0:左上, 1:右上, 2:右下, 3:左下
    cv::Rect m_cropRect;
};

#endif // WORKTHREAD_H
