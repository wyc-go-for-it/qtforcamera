#include "camera.h"

#include <OcrLite.h>
#include <QtWidgets>

#ifdef Q_OS_ANDROID
#include <androidUtils/localandroidutils.h>
#endif

#pragma execution_character_set("utf-8")

int main(int argc, char* argv[])
{

    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QCoreApplication::setAttribute(Qt::AA_NativeWindows);

    QApplication app(argc, argv);

    qRegisterMetaType<QVector<QPointF>>();
    qRegisterMetaType<QList<RecogResult>>();

#ifdef Q_OS_ANDROID
    if (!LocalAndroidUtils::checkAndRequestCameraPermission()) {
        return 1;
    }
#endif

#ifdef _WIN32
    SetConsoleOutputCP(65001); // 65001即 CP_UTF8
    SetConsoleCP(65001);
#endif

    Camera camera;
    camera.show();

    return app.exec();
};
