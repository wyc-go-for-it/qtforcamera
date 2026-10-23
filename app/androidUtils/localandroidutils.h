#ifndef LOCALANDROIDUTILS_H
#define LOCALANDROIDUTILS_H

#include <QAndroidJniObject>
#include <QString>

class LocalAndroidUtils final {
public:
    LocalAndroidUtils();
    static bool checkPermision(const QString& permision);
    static bool requestBlutoothPermissions();
    static int requestUsbPermissions(int vid, int pid, bool needFd = true);
    static QList<QPair<int, int>> usbDeivce();
    static bool usbWrite(int vid, int pid, const QByteArray& data);

    static bool checkAndRequestCameraPermission();

private:
    static bool requestPermissions(const QStringList& permissions);
    static QList<QAndroidJniObject> usbDeivceInternal();
    static QAndroidJniObject usbSeviceManager();

    static QString getAndroidId();
    static QString getImei();
};

#endif // LOCALANDROIDUTILS_H
