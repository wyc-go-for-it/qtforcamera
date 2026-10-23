#include "camera.h"
#include "qcoreapplication.h"

#include <QApplication>

#ifdef Q_OS_ANDROID
#include <androidUtils/localandroidutils.h>
#endif

int main(int argc, char* argv[])
{

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0))
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QCoreApplication::setAttribute(Qt::AA_NativeWindows);

    QApplication app(argc, argv);

    qRegisterMetaType<QVector<QPointF>>();
    qRegisterMetaType<QList<RecogResult>>();
    qRegisterMetaType<QVector<DetectObject>>();

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
