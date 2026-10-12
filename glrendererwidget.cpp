#include "glrendererwidget.h"

#include <QScroller>
#include <QThread>

GLRendererWidget::GLRendererWidget(QWindow* parent, QOpenGLWindow::UpdateBehavior behavior)
    : QOpenGLWindow(behavior, parent)
{
#ifdef Q_OS_ANDROID
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGLES);
    format.setVersion(3, 0);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    setFormat(format);
#endif

    m_rotationMatrix.setToIdentity();
}

GLRendererWidget::~GLRendererWidget()
{
    is_init = false;

    if (m_textures[0]) {
        this->glDeleteTextures(2, m_textures);
    }
}

void GLRendererWidget::setRotation(int angle)
{
    m_rotationAngle = angle;
    updateVertexPos();
    update();
}

void GLRendererWidget::updateFrame(const QVideoFrame& frame)
{
    if (!frame.isValid() || !is_init)
        return;

    QVideoFrame cloneFrame(frame);
    if (!cloneFrame.map(QAbstractVideoBuffer::ReadOnly))
        return;

    bool sizeChanged = (m_videoWidth != cloneFrame.width() || m_videoHeight != cloneFrame.height());
    m_videoWidth = cloneFrame.width();
    m_videoHeight = cloneFrame.height();

    if (sizeChanged) {
        updateVertexPos();
    }

    QVideoFrame::PixelFormat format = cloneFrame.pixelFormat();

    int bytesPerLine = cloneFrame.bytesPerLine();

    // 懒加载创建纹理
    if (!m_textures[0]) {
        this->glGenTextures(2, m_textures);
        for (int i = 0; i < 2; i++) {
            this->glBindTexture(GL_TEXTURE_2D, m_textures[i]);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }

    // 格式判定与纹理上传
    if (format == QVideoFrame::Format_YUYV || format == QVideoFrame::Format_YUV422P) {
        m_currentShaderType = ShaderType::YUYV;

        this->glActiveTexture(GL_TEXTURE0);
        this->glBindTexture(GL_TEXTURE_2D, m_textures[0]);

        this->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if (bytesPerLine > 0) {
            this->glPixelStorei(GL_UNPACK_ROW_LENGTH, bytesPerLine / 4);
        }

        this->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_videoWidth / 2, m_videoHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, cloneFrame.bits(0));
        this->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    } else if (format == QVideoFrame::Format_NV21) {
        m_currentShaderType = ShaderType::NV21;

        this->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // 上传 Y 平面 (单通道 GL_RED / GL_R8)
        this->glActiveTexture(GL_TEXTURE0);
        this->glBindTexture(GL_TEXTURE_2D, m_textures[0]);
        if (bytesPerLine > 0) {
            this->glPixelStorei(GL_UNPACK_ROW_LENGTH, bytesPerLine);
        }
        this->glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, m_videoWidth, m_videoHeight, 0, GL_RED, GL_UNSIGNED_BYTE, cloneFrame.bits(0));

        // 上传 UV 平面 (双通道 GL_RG / GL_RG8)
        this->glActiveTexture(GL_TEXTURE1);
        this->glBindTexture(GL_TEXTURE_2D, m_textures[1]);
        if (bytesPerLine > 0) {
            this->glPixelStorei(GL_UNPACK_ROW_LENGTH, bytesPerLine / 2); // 像素数跨度
        }
        this->glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, m_videoWidth / 2, m_videoHeight / 2, 0, GL_RG, GL_UNSIGNED_BYTE, cloneFrame.bits(1));

        this->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    } else {

        m_currentShaderType = ShaderType::RGB;

        const uchar* bits = cloneFrame.bits(0);

        if (bits) {
            this->glActiveTexture(GL_TEXTURE0);
            this->glBindTexture(GL_TEXTURE_2D, m_textures[0]);

            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            this->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            this->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

            // 计算跨度
            int strideInPixels = (bytesPerLine > 0) ? (bytesPerLine / 4) : m_videoWidth;

            // 兼容性保护：在标准 GLES 2.0 中不支持 GL_UNPACK_ROW_LENGTH
            // 只有当跨度不等于宽度时，才尝试使用此扩展，避免不必要的 GL 错误
            if (strideInPixels != m_videoWidth) {
                this->glPixelStorei(GL_UNPACK_ROW_LENGTH, strideInPixels);
            } else {
                this->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            }

            // 【修改点】：将 GL_RGBA8 替换为 GL_RGBA，解决 GLES 下的 invalid format 报错
            this->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_videoWidth, m_videoHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, bits);

            this->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        } else {
            // 备用兜底逻辑：若 bits(0) 为空才尝试转换
            QImage img = cloneFrame.image().convertToFormat(QImage::Format_RGBA8888);
            if (!img.isNull()) {
                this->glActiveTexture(GL_TEXTURE0);
                this->glBindTexture(GL_TEXTURE_2D, m_textures[0]);
                this->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                this->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                this->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(),
                    0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
            }
        }
    }

    cloneFrame.unmap();

    update();
}

void GLRendererWidget::updateCurrentObjects(const QVector<DetectObject>& objects)
{
    QMutexLocker l(&mlock);

    m_currentObjects = objects;
    update();
}

void GLRendererWidget::initializeGL()
{
    initializeOpenGLFunctions();
    this->glClearColor(0.8f, 0.8f, 0.8f, 1.0f);

    const char* vsrc = "attribute vec4 position;\n"
                       "attribute vec2 texCoord;\n"
                       "uniform mat4 rotationMatrix;\n"
                       "varying vec2 v_texCoord;\n"
                       "void main() {\n"
                       "  gl_Position = position;\n"
                       "  v_texCoord = vec2(rotationMatrix * vec4(texCoord, 0.0, 1.0));\n"
                       "}\n";

    const char* fsrcNV21 =
#ifdef Q_OS_ANDROID
        "precision mediump float;\n"
#endif
        "varying vec2 v_texCoord;\n"
        "uniform sampler2D sample_y;\n"
        "uniform sampler2D sample_uv;\n"
        "void main() {\n"
        "  float y = texture2D(sample_y, v_texCoord).r;\n"
        "  vec2 uv_color = texture2D(sample_uv, v_texCoord).rg;\n" // 改为 .rg
        "  float v = uv_color.r - 0.5;\n" // 第 1 个字节是 V
        "  float u = uv_color.g - 0.5;\n" // 第 2 个字节是 U (以前是 .a，现在改成 .g)
        "  gl_FragColor = vec4(\n"
        "    y + 1.402 * v,\n"
        "    y - 0.34414 * u - 0.71414 * v,\n"
        "    y + 1.772 * u,\n"
        "    1.0\n"
        "  );\n"
        "}\n";

    const char* fsrcYUYV =
#ifdef Q_OS_ANDROID
        "precision mediump float;\n"
#endif
        "varying vec2 v_texCoord;\n"
        "uniform sampler2D sample_yuyv;\n"
        "void main() {\n"
        "  vec4 yuyv = texture2D(sample_yuyv, v_texCoord);\n"
        "  float y = yuyv.r;\n"
        "  float u = yuyv.g - 0.5;\n"
        "  float v = yuyv.a - 0.5;\n"
        "  gl_FragColor = vec4(\n"
        "    y + 1.402 * v,\n"
        "    y - 0.344 * u - 0.714 * v,\n"
        "    y + 1.772 * u,\n"
        "    1.0\n"
        "  );\n"
        "}\n";

    const char* fsrcRGB =
#ifdef Q_OS_ANDROID
        "precision mediump float;\n"
#endif
        "varying vec2 v_texCoord;\n"
        "uniform sampler2D sample_rgb;\n"
        "void main() {\n"
        "  vec4 texColor = texture2D(sample_rgb, v_texCoord);\n"
        "  gl_FragColor = vec4(texColor.rgb, 1.0);\n"
        "}\n";

    const char* lineVertexShaderSource = R"(
    attribute vec2 aPos;
    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)";

    // 线框片段着色器 (Line Fragment Shader)

    const char* lineFragmentShaderSource = R"(
    precision mediump float;
    uniform vec4 uColor;
    void main() {
        gl_FragColor = uColor; // 自定义框的颜色，如红色 vec4(1.0, 0.0, 0.0, 1.0)
    }
)";

    initShader(m_programNV21, vsrc, fsrcNV21);
    m_programNV21.bind();
    m_programNV21.setUniformValue("sample_y", 0);
    m_programNV21.setUniformValue("sample_uv", 1);

    initShader(m_programYUYV, vsrc, fsrcYUYV);
    m_programYUYV.bind();
    m_programYUYV.setUniformValue("sample_yuyv", 0);

    initShader(m_programRGB, vsrc, fsrcRGB);
    m_programRGB.bind();
    m_programRGB.setUniformValue("sample_rgb", 0);

    initShader(m_lineProgram, lineVertexShaderSource, lineFragmentShaderSource);
    m_lineProgram.bind();

    is_init = true;
}

QRectF mapToWidgetCoords(const QRectF& objRect,
    const QSize& imageSize,
    const QSize& widgetSize)
{
    if (imageSize.isEmpty() || widgetSize.isEmpty())
        return QRectF();

    // 1. 计算缩放比例 (保持宽高比 Fit)
    float scaleX = static_cast<float>(widgetSize.width()) / imageSize.width();
    float scaleY = static_cast<float>(widgetSize.height()) / imageSize.height();
    float scale = std::min(scaleX, scaleY); // 选较小的缩放比，保证完整显示

    // 2. 计算图像渲染在 Widget 内部时的实际宽高
    float renderW = imageSize.width() * scale;
    float renderH = imageSize.height() * scale;

    // 3. 计算居中导致的黑边偏移量 (Offset)
    float offsetX = (widgetSize.width() - renderW) / 2.0f;
    float offsetY = (widgetSize.height() - renderH) / 2.0f;

    // 4. 将原始图像坐标点映射到 Widget 坐标系
    float x = objRect.x() * scale + offsetX;
    float y = objRect.y() * scale + offsetY;
    float w = objRect.width() * scale;
    float h = objRect.height() * scale;

    return QRectF(x, y, w, h);
}

void GLRendererWidget::paintGL()
{
    this->glClear(GL_COLOR_BUFFER_BIT);

    QOpenGLShaderProgram* activeProgram = nullptr;

    if (m_currentShaderType == ShaderType::NV21)
        activeProgram = &m_programNV21;
    else if (m_currentShaderType == ShaderType::YUYV)
        activeProgram = &m_programYUYV;
    else if (m_currentShaderType == ShaderType::RGB)
        activeProgram = &m_programRGB;

    if (activeProgram && m_textures[0]) {
        activeProgram->bind();

        this->glActiveTexture(GL_TEXTURE0);
        this->glBindTexture(GL_TEXTURE_2D, m_textures[0]);
        if (m_currentShaderType == ShaderType::NV21) {
            this->glActiveTexture(GL_TEXTURE1);
            this->glBindTexture(GL_TEXTURE_2D, m_textures[1]);
        }

        int rotLoc = activeProgram->uniformLocation("rotationMatrix");
        activeProgram->setUniformValue(rotLoc, m_rotationMatrix);

        static GLfloat tex[] = { 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        switch (((m_rotationAngle % 360) + 360) % 360) {
        case 0: {
            if (m_mirrored == QCamera::UnspecifiedPosition) {
                tex[0] = 0;
                tex[1] = 1;
                tex[2] = 1;
                tex[3] = 1;
                tex[4] = 0;
                tex[5] = 0;
                tex[6] = 1;
                tex[7] = 0;
            } else {
                tex[0] = 1;
                tex[1] = 1;
                tex[2] = 0;
                tex[3] = 1;
                tex[4] = 1;
                tex[5] = 0;
                tex[6] = 0;
                tex[7] = 0;
            }
        } break;
        case 90:
            tex[0] = 0;
            tex[1] = 1;
            tex[2] = 0;
            tex[3] = 0;
            tex[4] = 1;
            tex[5] = 1;
            tex[6] = 1;
            tex[7] = 0;
            break;
        case 180:
            tex[0] = 0;
            tex[1] = 0;
            tex[2] = 1;
            tex[3] = 0;
            tex[4] = 0;
            tex[5] = 1;
            tex[6] = 1;
            tex[7] = 1;
            break;
        case 270:
            tex[0] = 1;
            tex[1] = 0;
            tex[2] = 1;
            tex[3] = 1;
            tex[4] = 0;
            tex[5] = 0;
            tex[6] = 0;
            tex[7] = 1;
            break;
        }

        if (m_mirrored == QCamera::FrontFace) {
            // 前置：做一次 180° 交换（等价于同时翻 X 和翻 Y），
            // 用来抵消 Android HAL 对前置的镜像
            std::swap(tex[4], tex[2]);
            std::swap(tex[5], tex[3]);
            std::swap(tex[6], tex[0]);
            std::swap(tex[7], tex[1]);
        }

        this->glEnableVertexAttribArray(0);
        this->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, m_vertexPos);
        this->glEnableVertexAttribArray(1);
        this->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, tex);

        this->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        activeProgram->release();
    }

    {
        if (m_points.size() != 4)
            return;

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        QPolygonF poly(m_points);

        painter.setPen(QPen(Qt::red, 1, Qt::DotLine));
        painter.setBrush(QColor(255, 0, 0, 25));

        painter.drawRect(poly.boundingRect());

        painter.setPen(QPen(Qt::green, 1, Qt::SolidLine));
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(poly);

        for (int i = 0; i < 4; ++i) {
            painter.setPen(QPen(Qt::white, 2));
            painter.setBrush(i == m_draggedIndex ? Qt::yellow : Qt::green);
            painter.drawEllipse(m_points[i], 8, 8);
        }

        QFont font("Microsoft YaHei", 12); // Linux 下可换成 "WenQuanYi Micro Hei" 或 "Noto Sans CJK SC"
        painter.setFont(font);

        painter.setBrush(Qt::NoBrush);

        QMutexLocker l(&mlock);
        for (const auto& obj : m_currentObjects) {

            // 将模型输出坐标映射为当前 Widget 尺寸坐标
            QRectF rect = mapToWidgetCoords(obj.rect, m_actualImageSize, this->size());

            // 绘制矩形框
            painter.drawRect(rect);

            // 绘制分类与置信度标签
            QString label = QString("name %1:%2%").arg(obj.name).arg(int(obj.prob * 100));
            painter.drawText(rect.topLeft() + QPointF(5, -5), label);
        }
    }
}

void GLRendererWidget::resizeGL(int w, int h)
{
    this->glViewport(0, 0, w, h);
    updateVertexPos();
}

void GLRendererWidget::resizeEvent(QResizeEvent* event)
{
    QOpenGLWindow::resizeEvent(event);

    QSize oldSize = event->oldSize();
    QSize newSize = event->size();

    if (oldSize.width() <= 0 || oldSize.height() <= 0 || m_points.isEmpty()) {
        initDefaultPoints();
        return;
    }

    double scaleX = static_cast<double>(newSize.width()) / oldSize.width();
    double scaleY = static_cast<double>(newSize.height()) / oldSize.height();

    for (int i = 0; i < m_points.size(); ++i) {
        m_points[i].setX(m_points[i].x() * scaleX);
        m_points[i].setY(m_points[i].y() * scaleY);
    }

    update();
}

void GLRendererWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_draggedIndex = getHitPointIndex(event->pos());

        if (m_draggedIndex != -1) {
            m_isDraggingVertex = true;
        } else if (QPolygonF(m_points).containsPoint(event->pos(), Qt::OddEvenFill)) {
            m_isDraggingPolygon = true;
            m_lastMousePos = event->pos();
            setCursor(Qt::SizeAllCursor);
        }
    }
    QOpenGLWindow::mousePressEvent(event);
}

void GLRendererWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (m_isDraggingVertex && m_draggedIndex != -1) {

        QPointF pos = event->pos();
        pos.setX(qBound(0.0, pos.x(), (double)width()));
        pos.setY(qBound(0.0, pos.y(), (double)height()));

        m_points[m_draggedIndex] = pos;
        update();

        emit polygonChanged(calOriginalPoints());
    } else if (m_isDraggingPolygon) {
        QPointF delta = event->pos() - m_lastMousePos;
        bool canMove = true;
        for (const auto& pt : qAsConst(m_points)) {
            QPointF nextPt = pt + delta;
            if (nextPt.x() < 0 || nextPt.x() > width() || nextPt.y() < 0 || nextPt.y() > height()) {
                canMove = false;
                break;
            }
        }
        if (canMove) {
            for (auto& pt : m_points) {
                pt += delta;
            }
            m_lastMousePos = event->pos();
            update();

            emit polygonChanged(calOriginalPoints());
        }
    }

    QOpenGLWindow::mouseMoveEvent(event);
}

void GLRendererWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        if ((m_isDraggingVertex && m_draggedIndex != -1) || m_isDraggingPolygon) {
            savePoints();
        }

        m_isDraggingVertex = false;
        m_isDraggingPolygon = false;
        m_draggedIndex = -1;
        unsetCursor();
    }
    QOpenGLWindow::mouseReleaseEvent(event);
}

void GLRendererWidget::initDefaultPoints()
{

    m_points = loadPoints();

    if (m_points.isEmpty()) {
        m_points.resize(4);

        double w = width() > 0 ? width() : 640;
        double h = height() > 0 ? height() : 480;

        m_points[0] = QPointF(w * 0.2, h * 0.2); // 左上
        m_points[1] = QPointF(w * 0.8, h * 0.2); // 右上
        m_points[2] = QPointF(w * 0.85, h * 0.8); // 右下 (稍宽，模拟近大远小)
        m_points[3] = QPointF(w * 0.15, h * 0.8); // 左下
    }
}

QVector<QPointF> GLRendererWidget::loadPoints()
{
    QSettings settings("./config/hz_config", QSettings::Format::IniFormat);
    QVector<QPointF> points;

    int size = settings.beginReadArray("roi_points");
    for (int i = 0; i < size; ++i) {
        settings.setArrayIndex(i);
        QPointF pt = settings.value("point").toPointF();
        points.append(pt);
    }
    settings.endArray();

    qDebug() << "loadPoints:" << points;

    return points;
}

int GLRendererWidget::getHitPointIndex(const QPointF& mousePos)
{
    for (int i = 0; i < m_points.size(); ++i) {
        double dist = std::hypot(mousePos.x() - m_points[i].x(), mousePos.y() - m_points[i].y());
        if (dist <= 15.0) {
            return i;
        }
    }
    return -1;
}

QVector<QPointF> GLRendererWidget::calOriginalPoints()
{
    QVector<QPointF> imgPts;
    const QSize& imageSize = m_actualImageSize;
    if (m_points.empty() || imageSize.isEmpty())
        return imgPts;

    QSize viewfinderSize = this->size();

    QSize scaledSize = imageSize.scaled(viewfinderSize, Qt::KeepAspectRatio);
    double targetX = (viewfinderSize.width() - scaledSize.width()) / 2.0;
    double targetY = (viewfinderSize.height() - scaledSize.height()) / 2.0;

    double scaleX = static_cast<double>(imageSize.width()) / scaledSize.width();
    double scaleY = static_cast<double>(imageSize.height()) / scaledSize.height();

    for (const auto& pt : qAsConst(m_points)) {

        double imgX = (pt.x() - targetX) * scaleX;
        double imgY = (pt.y() - targetY) * scaleY;

        imgX = std::clamp(imgX, 0.0, static_cast<double>(imageSize.width() - 1));
        imgY = std::clamp(imgY, 0.0, static_cast<double>(imageSize.height() - 1));

        imgPts.push_back(QPointF(imgX, imgY));
    }

    return imgPts;
}

void GLRendererWidget::savePoints()
{
    QSettings settings("./config/hz_config", QSettings::Format::IniFormat);

    settings.beginWriteArray("roi_points");
    for (int i = 0; i < m_points.size(); ++i) {
        settings.setArrayIndex(i);
        settings.setValue("point", m_points[i]);
    }
    settings.endArray();

    qDebug() << "savePoints:" << m_points;
}

void GLRendererWidget::updateVertexPos()
{
    if (m_videoWidth <= 0 || m_videoHeight <= 0 || width() <= 0 || height() <= 0)
        return;

    float targetW = m_videoWidth;
    float targetH = m_videoHeight;

    if (m_rotationAngle == 90 || m_rotationAngle == 270) {
        std::swap(targetW, targetH);
    }

    float videoAspect = targetW / targetH;
    float widgetAspect = static_cast<float>(width()) / static_cast<float>(height());

    float xRatio = 1.0f;
    float yRatio = 1.0f;

    if (videoAspect > widgetAspect) {
        yRatio = widgetAspect / videoAspect;
    } else {
        xRatio = videoAspect / widgetAspect;
    }

    m_vertexPos[0] = -xRatio;
    m_vertexPos[1] = -yRatio;
    m_vertexPos[2] = xRatio;
    m_vertexPos[3] = -yRatio;
    m_vertexPos[4] = -xRatio;
    m_vertexPos[5] = yRatio;
    m_vertexPos[6] = xRatio;
    m_vertexPos[7] = yRatio;
}

void GLRendererWidget::initShader(QOpenGLShaderProgram& program, const char* vsrc, const char* fsrc)
{
    program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsrc);
    program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsrc);
    program.bindAttributeLocation("position", 0);
    program.bindAttributeLocation("texCoord", 1);
    program.link();
}

std::vector<float> GLRendererWidget::generateBoxVertices()
{
    std::vector<float> vertices;

    auto imgSize = m_actualImageSize;
    auto widgetSize = this->size();

    if (imgSize.isEmpty() || widgetSize.isEmpty())
        return vertices;

    // 1. 计算 KeepAspectRatio 保持宽高比下的 Scale 和 Offset
    float scaleX = static_cast<float>(widgetSize.width()) / imgSize.width();
    float scaleY = static_cast<float>(widgetSize.height()) / imgSize.height();
    float scale = std::min(scaleX, scaleY);

    float renderW = imgSize.width() * scale;
    float renderH = imgSize.height() * scale;
    float offsetX = (widgetSize.width() - renderW) / 2.0f;
    float offsetY = (widgetSize.height() - renderH) / 2.0f;

    float W = widgetSize.width();
    float H = widgetSize.height();

    for (const auto& obj : m_currentObjects) {
        // 还原到 Widget 像素坐标系 (左上角 0,0)
        float x0 = obj.rect.x() * scale + offsetX;
        float y0 = obj.rect.y() * scale + offsetY;
        float x1 = x0 + obj.rect.width() * scale;
        float y1 = y0 + obj.rect.height() * scale;

        // 2. 转换像素坐标系 到 OpenGL NDC 坐标系 (-1 到 1，Y轴翻转)
        float ndc_x0 = (x0 / W) * 2.0f - 1.0f;
        float ndc_y0 = 1.0f - (y0 / H) * 2.0f; // Y轴反向
        float ndc_x1 = (x1 / W) * 2.0f - 1.0f;
        float ndc_y1 = 1.0f - (y1 / H) * 2.0f;

        // 生成矩形 4 条边的线段顶点 (使用 GL_LINES，共 4 组点/8 个顶点)
        // 上边
        vertices.push_back(ndc_x0);
        vertices.push_back(ndc_y0);
        vertices.push_back(ndc_x1);
        vertices.push_back(ndc_y0);
        // 右边
        vertices.push_back(ndc_x1);
        vertices.push_back(ndc_y0);
        vertices.push_back(ndc_x1);
        vertices.push_back(ndc_y1);
        // 下边
        vertices.push_back(ndc_x1);
        vertices.push_back(ndc_y1);
        vertices.push_back(ndc_x0);
        vertices.push_back(ndc_y1);
        // 左边
        vertices.push_back(ndc_x0);
        vertices.push_back(ndc_y1);
        vertices.push_back(ndc_x0);
        vertices.push_back(ndc_y0);
    }

    return vertices;
}

void GLRendererWidget::drawYoloBoxes()
{
    // 生成所有框的 NDC 顶点数组
    std::vector<float> lineVertices = generateBoxVertices();
    if (lineVertices.empty())
        return;

    m_lineProgram.bind();

    // 设置线条颜色 (例如红色)
    m_lineProgram.setUniformValue("uColor", QVector4D(1.0f, 0.0f, 0.0f, 1.0f));

    // 设置线宽
    glLineWidth(2.0f);

    int posLocation = m_lineProgram.attributeLocation("aPos");
    m_lineProgram.enableAttributeArray(posLocation);
    m_lineProgram.setAttributeArray(posLocation, GL_FLOAT, lineVertices.data(), 2);

    // 一次 GPU Call 批量绘制所有线段
    glDrawArrays(GL_LINES, 0, lineVertices.size() / 2);

    m_lineProgram.disableAttributeArray(posLocation);
    m_lineProgram.release();
}
