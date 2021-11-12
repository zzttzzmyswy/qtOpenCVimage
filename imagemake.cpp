#include "imagemake.h"

imageMake::imageMake(QObject *parent) : QObject(parent) {}

void imageMake::makeHistogram(QImage inimage, QCustomPlot *pchart,
                              QImage *outimage) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
    if (outimage != NULL)
      (*outimage) = grayimage.copy();
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;

  int nWidth = inimage.width();
  int nHeight = inimage.height();
  QVector<double> vecX;
  QVector<double> vecY(256, 0); // init Y data with 0;

  int i = 0;
  while (256 != i) {
    vecX.append(i);
    ++i;
  }

  for (int j = 0; j < nHeight; j++) {
    for (int k = 0; k < nWidth; k++) {
      quint8 nIndex = quint8(grayimage.bits()[j * nHeight + k]);
      vecY[nIndex] = vecY.at(nIndex) + 1;
    }
  }

  double yMax = 0;
  for (int j = 0; j < 256; j++) {
    if (yMax < vecY.at(j))
      yMax = vecY.at(j);
  }

  //建表
  pchart->xAxis->setLabel(tr("Grayscale"));
  pchart->yAxis->setLabel(tr("Number"));
  pchart->xAxis->setRange(0, 255);
  pchart->yAxis->setRange(0, yMax);
  pchart->graph(0)->setData(vecX, vecY);
  //    m_pChart->graph(0)->setPen(QPen(Qt::red));
  pchart->replot();

  //  chartview->setChart(chart);
  //  chartview->setRenderHint(QPainter::Antialiasing);
  //  chartview->setVisible(true);
}

void imageMake::makeHistogramAveraging(QImage inimage, QCustomPlot *pchart,
                                       QImage *outimage) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
    if (outimage != NULL)
      (*outimage) = grayimage.copy();
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;

  /* qimage 转 opencv Mat */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  /* 利用函数 equalizeHist执行直方图均值化 */
  equalizeHist(mat0, mat1);
  QImage outImage;
  /* opencv Mat 转 qimage */
  outImage = MatToQImage(mat1).copy();
  if (outimage != NULL)
    (*outimage) = outImage.copy();
  makeHistogram(outImage, pchart, NULL);
}

void imageMake::makeBinarization(QImage inimage, QImage *outimage,
                                 quint8 threshold) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 获取二值化后的信息 */
  int width = grayimage.width();
  int height = grayimage.height();
  int bytePerLine = grayimage.bytesPerLine(); // 每一行的字节数
  quint8 *data = grayimage.bits();
  quint8 *binarydata = new unsigned char[bytePerLine * height];
  quint8 temp_g = 0;
  /* 开始计算所有点数 */
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      temp_g = *(data + i * bytePerLine + j);
      if (temp_g >= threshold) {
        binarydata[i * bytePerLine + j] = 0xFF;
      } else {
        binarydata[i * bytePerLine + j] = 0x00;
      }
    }
  }
  if (outimage != NULL)
    /* 输出图像 */
    (*outimage) = QImage(binarydata, width, height, bytePerLine,
                         QImage::Format_Grayscale8)
                      .copy();
  delete[] binarydata;
}

void imageMake::makeAdaptiveThreshold(QImage inimage, QImage *outimage1,
                                      QImage *outimage2, quint8 area,
                                      quint8 numC) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;

  cv::Mat mat0, mat1, mat2;
  mat0 = QImageToMat(grayimage);
  adaptiveThreshold(mat0, mat1, 0xff, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
                    area, numC);
  adaptiveThreshold(mat0, mat2, 0xff, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                    area, numC);
  if (outimage1 != NULL)
    (*outimage1) = MatToQImage(mat1).copy();
  if (outimage2 != NULL)
    (*outimage2) = MatToQImage(mat2).copy();
}

void imageMake::makeQtsu(QImage inimage, QImage *outimage) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  threshold(mat0, mat1, 0, 0xff, THRESH_BINARY + THRESH_OTSU);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeSobel(QImage inimage, QImage *outimage, quint8 k) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;
  mat0 = QImageToMat(grayimage);
  /* 将原灰度图进行高斯降噪 */
  GaussianBlur(mat0, mat0, Size(3, 3), 0, 0, BORDER_DEFAULT);
  /*
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize = 3,
                         double scale = 1, double delta = 0,
                         int borderType = BORDER_DEF
    输入矩阵
    输出矩阵
    像素计算深度
    x方向求导阶数
    y方向求导阶数
    核大小
    后面默认
*/

  /* 求 X方向梯度 */
  Sobel(mat0, grad_x, CV_16S, 1, 0, k, 1, 0, BORDER_DEFAULT);
  /* 求 Y方向梯度 */
  Sobel(mat0, grad_y, CV_16S, 0, 1, k, 1, 0, BORDER_DEFAULT);

  /* 将中间结果转换到 CV_8U */
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  /* 叠加x方向和y方向的数据 */
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, mat1);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeScharr(QImage inimage, QImage *outimage) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;
  mat0 = QImageToMat(grayimage);
  /* 将原灰度图进行高斯降噪 */
  GaussianBlur(mat0, mat0, Size(3, 3), 0, 0, BORDER_DEFAULT);
  /*
CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,
                          int dx, int dy, double scale = 1, double delta = 0,
                          int borderType = BORDER_DEFAULT );
    输入矩阵
    输出矩阵
    像素计算深度
    x方向求导阶数
    y方向求导阶数
    后面默认

    默认核为3
*/

  /* 求 X方向梯度 */
  Scharr(mat0, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
  /* 求 Y方向梯度 */
  Scharr(mat0, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);

  /* 将中间结果转换到 CV_8U */
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  /* 叠加x方向和y方向的数据 */
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, mat1);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeLaplace(QImage inimage, QImage *outimage, quint8 k) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1, mat2;
  mat0 = QImageToMat(grayimage);
  /* 将原灰度图进行高斯降噪 */
  GaussianBlur(mat0, mat0, Size(3, 3), 0, 0, BORDER_DEFAULT);
  /*
CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                             int ksize = 1, double scale = 1, double delta = 0,
                             int borderType = BORDER_DEFAULT );
    输入矩阵
    输出矩阵
    像素计算深度
    后面默认
*/
  /* 执行Laplace 算子 */
  Laplacian(mat0, mat1, CV_16S, k, 1, 8, BORDER_DEFAULT);

  /* 将中间结果转换到 CV_8U */
  convertScaleAbs(mat1, mat2);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat2).copy();
}

void imageMake::makeCanny(QImage inimage, QImage *outimage, quint8 threshold) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  /* 将原灰度图进行高斯降噪 */
  GaussianBlur(mat0, mat0, Size(3, 3), 0, 0, BORDER_DEFAULT);
  /*
CV_EXPORTS_W void Canny( InputArray image, OutputArray edges,
                         double threshold1, double threshold2,
                         int apertureSize = 3, bool L2gradient = false );

    输入矩阵
    输出矩阵
    低阈值
    高阈值（低阈值的2-3倍）
    内部sodel算子核的大小
    是否使用精确的公式计算梯度幅值和方向
*/
  /* 执行Laplace 算子 */
  Canny(mat0, mat1, threshold, threshold * 3, 3, true);

  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeHoughLines(QImage inimage, QImage *outimage,
                               quint32 threshold) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  cvtColor(mat0, mat1, CV_GRAY2BGR);
  std::vector<Vec2f> lines;
  /*
  CV_EXPORTS_W void HoughLines( InputArray image, OutputArray lines,
                                double rho, double theta, int threshold,
                                double srn = 0, double stn = 0,
                                double min_theta = 0, double max_theta = CV_PI
  );
  输入矩阵
  输出矩阵
  单位分辨率，默认1像素
  参数角分辨率，默认1度 CV_PI/180
  判断是否直线的阈值
  后面的值默认
  */

  HoughLines(mat0, lines, 0.5, CV_PI / 180, threshold, 0, 0, 0, CV_PI);
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    /*
CV_EXPORTS_W void line(InputOutputArray img, Point pt1, Point pt2, const Scalar&
color, int thickness = 1, int lineType = LINE_8, int shift = 0);
*/
    line(mat1, pt1, pt2, Scalar(0, 0, 255), 3, 0);
  }
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeHoughLinesP(QImage inimage, QImage *outimage,
                                quint32 threshold) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  cvtColor(mat0, mat1, CV_GRAY2BGR);
  std::vector<Vec4i> lines;
  /*
CV_EXPORTS_W void HoughLinesP( InputArray image, OutputArray lines,
                               double rho, double theta, int threshold,
                               double minLineLength = 0, double maxLineGap = 0
);
  );
  输入矩阵
  输出矩阵
  单位分辨率，默认1像素
  参数角分辨率，默认1度 CV_PI/180
  判断是否直线的阈值
  后面的值默认
  */

  HoughLinesP(mat0, lines, 0.5, CV_PI / 180, threshold, 20, 10);
  for (size_t i = 0; i < lines.size(); i++) {
    Vec4i l = lines[i];
    /*
CV_EXPORTS_W void line(InputOutputArray img, Point pt1, Point pt2, const Scalar&
color, int thickness = 1, int lineType = LINE_8, int shift = 0);
*/
    line(mat1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 0);
  }
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeMeanFilter(QImage inimage, QImage *outimage, quint32 k) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  blur(mat0, mat1, Size(k, k));
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeGaussianFilter(QImage inimage, QImage *outimage,
                                   quint32 k) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  GaussianBlur(mat0, mat1, Size(k, k), 0, 0);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeMedianFilter(QImage inimage, QImage *outimage, quint32 k) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  /*
CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );
    */
  medianBlur(mat0, mat1, k);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

void imageMake::makeBilateralFilter(QImage inimage, QImage *outimage, quint32 k,
                                    quint32 sigmaColor, quint32 sigmaSpace) {
  if (inimage.isNull())
    return;
  /* 判断是否为灰度图，若不是是灰度图则转化 */
  QImage grayimage;
  if (!inimage.allGray()) {
    grayimage = inimage.convertToFormat(QImage::Format_Grayscale8);
  } else {
    grayimage = inimage.copy();
  }
  if (grayimage.isNull())
    return;
  /* 创建 grad_x 和 grad_y 矩阵 */
  cv::Mat mat0, mat1;
  mat0 = QImageToMat(grayimage);
  /*
CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,
                                   double sigmaColor, double sigmaSpace,
                                   int borderType = BORDER_DEFAULT );
    */
  bilateralFilter(mat0, mat1, k, sigmaColor, sigmaSpace);
  if (outimage != NULL)
    (*outimage) = MatToQImage(mat1).copy();
}

cv::Mat imageMake::QImageToMat(QImage image) {
  cv::Mat mat;
  switch (image.format()) {
  case QImage::Format_ARGB32:
  case QImage::Format_RGB32:
  case QImage::Format_ARGB32_Premultiplied:
    mat = cv::Mat(image.height(), image.width(), CV_8UC4,
                  (void *)image.constBits(), image.bytesPerLine());
    break;
  case QImage::Format_RGB888:
    mat = cv::Mat(image.height(), image.width(), CV_8UC3,
                  (void *)image.constBits(), image.bytesPerLine());
    cv::cvtColor(mat, mat, CV_BGR2RGB);
    break;
  case QImage::Format_Indexed8:
  case QImage::Format_Grayscale8:
    mat = cv::Mat(image.height(), image.width(), CV_8UC1,
                  (void *)image.constBits(), image.bytesPerLine());
    break;
  default:
    return mat;
  }
  return mat;
}

QImage imageMake::MatToQImage(const cv::Mat &mat) {
  // 8-bits unsigned, NO. OF CHANNELS = 1
  if (mat.type() == CV_8UC1) {
    QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
    // Set the color table (used to translate colour indexes to qRgb values)
    image.setColorCount(256);
    for (int i = 0; i < 256; i++) {
      image.setColor(i, qRgb(i, i, i));
    }
    // Copy input Mat
    uchar *pSrc = mat.data;
    for (int row = 0; row < mat.rows; row++) {
      uchar *pDest = image.scanLine(row);
      memcpy(pDest, pSrc, mat.cols);
      pSrc += mat.step;
    }
    return image;
  }
  // 8-bits unsigned, NO. OF CHANNELS = 3
  else if (mat.type() == CV_8UC3) {
    // Copy input Mat
    const uchar *pSrc = (const uchar *)mat.data;
    // Create QImage with same dimensions as input Mat
    QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    return image.rgbSwapped();
  } else if (mat.type() == CV_8UC4) {
    qDebug() << "CV_8UC4";
    // Copy input Mat
    const uchar *pSrc = (const uchar *)mat.data;
    // Create QImage with same dimensions as input Mat
    QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
    return image.copy();
  } else {
    qDebug() << "ERROR: Mat could not be converted to QImage.";
    return QImage();
  }
}
