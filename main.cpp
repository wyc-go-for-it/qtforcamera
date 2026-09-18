#include "camera.h"

#include <QtWidgets>

#ifdef Q_OS_ANDROID
#include <androidUtils/localandroidutils.h>
#endif

int main(int argc, char* argv[])
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QCoreApplication::setAttribute(Qt::AA_NativeWindows);

    QApplication app(argc, argv);

    qRegisterMetaType<QVector<QPointF>>();

#ifdef Q_OS_ANDROID
    if (!LocalAndroidUtils::checkAndRequestCameraPermission()) {
        return 1;
    }
#endif

    Camera camera;
    camera.show();

    return app.exec();
};
