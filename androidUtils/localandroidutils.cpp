#include "localandroidutils.h"
#include <QAndroidJniObject>
#include <QtAndroid>

#include <QAndroidIntent>
#include <QAndroidJniEnvironment>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QUuid>

LocalAndroidUtils::LocalAndroidUtils() { }

bool LocalAndroidUtils::checkAndRequestCameraPermission()
{
    const QString permission = "android.permission.CAMERA";
    return requestPermissions({ permission });
}

bool LocalAndroidUtils::checkPermision(const QString& permision)
{
    if (QtAndroid::androidSdkVersion() >= 30) {
        jboolean isExternalStorageManager = QAndroidJniObject::callStaticMethod<jboolean>("android/os/Environment", "isExternalStorageManager", "()Z");
        if (!isExternalStorageManager) {
            QAndroidIntent intent("android.settings.MANAGE_ALL_FILES_ACCESS_PERMISSION");
            QtAndroid::startActivity(intent.handle(), 0);
        }
    } else {
        if (QtAndroid::checkPermission(permision) == QtAndroid::PermissionResult::Denied) {
            return requestPermissions({ permision });
        }
    }
    return true;
}

bool LocalAndroidUtils::requestPermissions(const QStringList& permissions)
{
    const auto rMap = QtAndroid::requestPermissionsSync(permissions);
    foreach (const auto& v, permissions) {
        if (rMap[v] == QtAndroid::PermissionResult::Denied) {
            return false;
        }
    }
    return true;
}

bool LocalAndroidUtils::requestBlutoothPermissions()
{
    qDebug() << "androidSdkVersion:" << QtAndroid::androidSdkVersion();

    if (QtAndroid::androidSdkVersion() > 30) {
        return requestPermissions({ "android.permission.ACCESS_FINE_LOCATION", "android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_ADVERTISE", "android.permission.BLUETOOTH_CONNECT" });
    } else
        return requestPermissions({ "android.permission.ACCESS_FINE_LOCATION", "android.permission.ACCESS_COARSE_LOCATION" });
}

QList<QAndroidJniObject> LocalAndroidUtils::usbDeivceInternal()
{
    QAndroidJniObject usbManger = usbSeviceManager();

    QAndroidJniObject usbDeviceLstMap = usbManger.callObjectMethod("getDeviceList", "()Ljava/util/HashMap;");

    QAndroidJniObject keySet = usbDeviceLstMap.callObjectMethod("keySet", "()Ljava/util/Set;");
    QAndroidJniObject iter = keySet.callObjectMethod("iterator", "()Ljava/util/Iterator;");

    QAndroidJniObject key;
    QAndroidJniObject deviceObj;

    QList<QAndroidJniObject> deviceLst;
    while (iter.callMethod<jboolean>("hasNext", "()Z")) {
        key = iter.callObjectMethod("next", "()Ljava/lang/Object;");
        deviceObj = usbDeviceLstMap.callObjectMethod("get", "(Ljava/lang/Object;)Ljava/lang/Object;", key.object());
        if (deviceObj.isValid()) {
            deviceLst.append(deviceObj);
        }
    }

    return deviceLst;
}

QAndroidJniObject LocalAndroidUtils::usbSeviceManager()
{
    QAndroidJniObject usbSeviceName = QAndroidJniObject::getStaticObjectField("android/content/Context", "USB_SERVICE", "Ljava/lang/String;");
    QAndroidJniObject context = QtAndroid::androidContext();
    QAndroidJniObject usbManger = context.callObjectMethod("getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;", usbSeviceName.object<jstring>());

    return usbManger;
}

QString LocalAndroidUtils::getAndroidId()
{
    QAndroidJniObject context = QtAndroid::androidContext();
    QAndroidJniObject contentResolver = context.callObjectMethod(
        "getContentResolver", "()Landroid/content/ContentResolver;");

    QAndroidJniObject androidId = QAndroidJniObject::callStaticObjectMethod(
        "android/provider/Settings$Secure",
        "getString",
        "(Landroid/content/ContentResolver;Ljava/lang/String;)Ljava/lang/String;",
        contentResolver.object(),
        QAndroidJniObject::fromString("android_id").object());

    return androidId.toString();
}

QString LocalAndroidUtils::getImei()
{

    QtAndroid::PermissionResultMap result = QtAndroid::requestPermissionsSync(QStringList({ "android.permission.READ_PHONE_STATE" }));

    if (result["android.permission.READ_PHONE_STATE"] == QtAndroid::PermissionResult::Granted) {
        // 需要 READ_PHONE_STATE 权限
        QAndroidJniObject context = QtAndroid::androidContext();
        QAndroidJniObject telephonyService = context.callObjectMethod(
            "getSystemService",
            "(Ljava/lang/String;)Ljava/lang/Object;",
            QAndroidJniObject::fromString("phone").object());

        QAndroidJniObject imei = telephonyService.callObjectMethod(
            "getDeviceId",
            "()Ljava/lang/String;");

        return imei.toString();
    }

    return {};
}

int LocalAndroidUtils::requestUsbPermissions(int vid, int pid, bool needFd)
{
    jint fd = -1;

    const auto& lst = usbDeivceInternal();

    foreach (const auto& deviceObj, lst) {

        jint vidObj = deviceObj.callMethod<jint>("getVendorId", "()I");
        jint pidObj = deviceObj.callMethod<jint>("getProductId", "()I");

        if (vidObj == vid && pidObj == pid) {
            QAndroidJniObject usbManger = usbSeviceManager();

            if (!usbManger.callMethod<jboolean>("hasPermission", "(Landroid/hardware/usb/UsbDevice;)Z", deviceObj.object())) {
                QAndroidJniObject permissionIntent = QAndroidJniObject::callStaticObjectMethod(
                    "android/app/PendingIntent", "getBroadcast", "(Landroid/content/Context;ILandroid/content/Intent;I)Landroid/app/PendingIntent;",
                    QtAndroid::androidContext().object(), 0, QAndroidIntent("com.wyc.cloudapp.USB_PERMISSION").handle().object(), 0);

                usbManger.callMethod<void>("requestPermission", "(Landroid/hardware/usb/UsbDevice;Landroid/app/PendingIntent;)V", deviceObj.object(), permissionIntent.object());

                fd = -2;
            } else if (needFd) {
                QAndroidJniObject usbDeviceConnection = usbManger.callObjectMethod("openDevice", "(Landroid/hardware/usb/UsbDevice;)Landroid/hardware/usb/UsbDeviceConnection;", deviceObj.object());
                fd = usbDeviceConnection.callMethod<jint>("getFileDescriptor", "()I");
            }
            break;
        }
    }

    return fd;
}

QList<QPair<int, int>> LocalAndroidUtils::usbDeivce()
{
    QList<QPair<int, int>> deviceLst;
    const auto& lst = usbDeivceInternal();
    foreach (const auto& dev, lst) {

        jint deviceClass = dev.callMethod<jint>("getDeviceClass", "()I");
        if (deviceClass != 7) {
            jint vidObj = dev.callMethod<jint>("getVendorId", "()I");
            jint pidObj = dev.callMethod<jint>("getProductId", "()I");
            deviceLst.append(qMakePair(vidObj, pidObj));
        }
    }
    return deviceLst;
}

bool LocalAndroidUtils::usbWrite(int vid, int pid, const QByteArray& data)
{
    const auto& lst = usbDeivceInternal();

    foreach (const auto& deviceObj, lst) {

        jint vidObj = deviceObj.callMethod<jint>("getVendorId", "()I");
        jint pidObj = deviceObj.callMethod<jint>("getProductId", "()I");

        if (vidObj == vid && pidObj == pid) {
            QAndroidJniObject usbInterface = deviceObj.callObjectMethod("getInterface", "(I)Landroid/hardware/usb/UsbInterface;", 0);

            jint endpointCount = usbInterface.callMethod<jint>("getEndpointCount", "()I");

            QAndroidJniObject endpoint;

            for (int i = 0; i < endpointCount; i++) {
                endpoint = usbInterface.callObjectMethod("getEndpoint", "(I)Landroid/hardware/usb/UsbEndpoint;", i);

                jint direction = endpoint.callMethod<jint>("getDirection", "()I");
                if (direction == 0) {
                    QAndroidJniObject usbManger = usbSeviceManager();
                    QAndroidJniObject usbDeviceConnection = usbManger.callObjectMethod("openDevice", "(Landroid/hardware/usb/UsbDevice;)Landroid/hardware/usb/UsbDeviceConnection;", deviceObj.object());

                    jboolean claimInterface = usbDeviceConnection.callMethod<jboolean>("claimInterface", "(Landroid/hardware/usb/UsbInterface;Z)Z", usbInterface.object(), true);

                    if (claimInterface) {

                        int size = data.size();

                        QAndroidJniEnvironment env;
                        jbyteArray d = env->NewByteArray(size);
                        env->SetByteArrayRegion(d, 0, size, (const jbyte*)data.constData());

                        jint cnt = usbDeviceConnection.callMethod<jint>("bulkTransfer", "(Landroid/hardware/usb/UsbEndpoint;[BII)I",
                            endpoint.object(), d, size, 100);

                        env->DeleteLocalRef(d);

                        usbDeviceConnection.callMethod<jboolean>("releaseInterface", "(Landroid/hardware/usb/UsbInterface;)Z", usbInterface.object());
                        usbDeviceConnection.callMethod<void>("close", "()V");

                        return cnt == size;
                    }
                    break;
                }
            }
            break;
        }
    }

    return false;
}
