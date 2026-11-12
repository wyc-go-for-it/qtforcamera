#include "facedialog.h"
#include "filepickerdialog.h"
#include "ui_facedialog.h"
#include <ImageUtils.h>
#include <QCameraInfo>
#include <QFileDialog>
#include <QMessageBox>
#include <QVideoProbe>

FaceDialog::FaceDialog(QWidget* parent)
    : QDialog(parent)
    , ui(new Ui::FaceDialog)
    , mWorkThread(new FaceWorkThread())
{
    ui->setupUi(this);

    connect(mWorkThread.data(), &FaceWorkThread::error, this, [this](const QString& error) {
        ui->msg->setText(error);
    });

    connect(mWorkThread.data(), &FaceWorkThread::info, this, [this](const QString& error) {
        ui->info->setText(error);
    });
}

FaceDialog::~FaceDialog()
{
    delete ui;
}

void FaceDialog::updateGLSurface(QCamera* camera, const QCameraInfo& cameraInfo)
{
    this->m_camera = camera;

    m_glWindow = new GLRendererWidget();
    m_glWindow->setObjectName("face");

    m_glWindow->setMirrored(cameraInfo.position());
    m_glWindow->setRotation(cameraInfo.orientation());

    m_glContainer.reset(QWidget::createWindowContainer(m_glWindow));

    m_glContainer->setObjectName("m_glContainer");
    m_glContainer->installEventFilter(this);
    m_glContainer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_glContainer->setAttribute(Qt::WA_AcceptTouchEvents, true);
    ui->viewfinderPage->layout()->setMargin(0);
    ui->viewfinderPage->layout()->addWidget(m_glContainer.data());

    connect(mWorkThread.data(), &FaceWorkThread::detected, m_glWindow, &GLRendererWidget::updateCurrentObjects, Qt::QueuedConnection);

    connect(mWorkThread.data(), &FaceWorkThread::captured, this, [this](const QImage& frame, const QImage& reg, const QImage& alinged) {
        qreal dpi = ui->contrast_img->devicePixelRatioF();

        auto scaledFrame = frame.scaled(ui->contrast_img->size() * dpi,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        scaledFrame.setDevicePixelRatio(dpi);
        ui->contrast_img->setPixmap(QPixmap::fromImage(scaledFrame));

        auto scaledDiff = alinged.scaled(ui->aligned_img->size() * dpi,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        scaledDiff.setDevicePixelRatio(dpi);
        ui->aligned_img->setPixmap(QPixmap::fromImage(scaledDiff));

        auto regFrame = reg.scaled(ui->contrast_img->size() * dpi,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        regFrame.setDevicePixelRatio(dpi);
        ui->reg_img->setPixmap(QPixmap::fromImage(regFrame));
    });

    auto* probe = new QVideoProbe(this);
    if (probe->setSource(m_camera.data())) {
        probe->connect(probe, &QVideoProbe::videoFrameProbed, this, [this](const QVideoFrame& frame) {
            if (!isFirstFrame) {
                qDebug() << "viewfinderSettings Formats:" << m_camera->viewfinderSettings().pixelFormat();
                qDebug() << "supportedViewfinderPixelFormats:" << m_camera->supportedViewfinderPixelFormats();
                qDebug() << "QVideoFrame Formats:" << frame.pixelFormat();
                qDebug() << "QVideoFrame size:" << frame.size();

                isFirstFrame = true;
                m_glWindow->updateImageSize(frame.size());
            }
            m_glWindow->updateFrame(frame);

            mWorkThread->onVideoFrameChanged(frame);
        });
    }

    connect(m_glWindow, &GLRendererWidget::polygonChanged, mWorkThread.data(), &FaceWorkThread::onUpdateVertex);

    mWorkThread->init();
}

void FaceDialog::on_register_face_clicked()
{
    mWorkThread->onRegisteredFace(true, ui->name->text());
    ui->name->clear();
}

void FaceDialog::on_query_face_clicked()
{
    mWorkThread->onQueriedFace(true, ui->name->text());
}

void FaceDialog::on_extract_feature_clicked()
{
    QMessageBox msgBox(this);
    msgBox.setText("是否生成测试数据");
    msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    if (msgBox.exec() == QMessageBox::Ok) {
        mWorkThread->onExtractFeature();
    }
}

void FaceDialog::on_select_file_clicked()
{

    FilePickerDialog dlg(this);
    dlg.exec();

    mWorkThread->onQueriedFace(true, dlg.selectedPath());
}

void FaceDialog::on_test_clicked()
{
    mWorkThread->onRandomTest();
}
