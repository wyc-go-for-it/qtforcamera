#ifndef FACEDIALOG_H
#define FACEDIALOG_H

#include "CustomGLSurface.h"
#include "face/faceengine.h"
#include <FaceWorkThread.h>
#include <GLRendererWidget.h>
#include <QDialog>
#include <QPointer>

namespace Ui {
class FaceDialog;
}

class FaceDialog : public QDialog {
    Q_OBJECT

public:
    explicit FaceDialog(QWidget* parent = nullptr);
    ~FaceDialog();

    void updateGLSurface(QCamera* camera, const QCameraInfo& cameraInfo);

private slots:
    void on_register_face_clicked();

    void on_query_face_clicked();

    void on_extract_feature_clicked();

    void on_select_file_clicked();

    void on_test_clicked();

private:
    Ui::FaceDialog* ui;

    QPointer<QCamera> m_camera;
    QScopedPointer<QWidget> m_glContainer;
    GLRendererWidget* m_glWindow;

    FaceEngine m_faceEngine;

    QScopedPointer<FaceWorkThread> mWorkThread;

    bool isFirstFrame = false;
};

#endif // FACEDIALOG_H
