#ifndef IMAGEMAKE_H
#define IMAGEMAKE_H

#include <QBarCategoryAxis>
#include <QBarSeries>
#include <QBarSet>
#include <QChart>
#include <QImage>
#include <QObject>
#include <QValueAxis>
#include <qcustomplot/qcustomplot.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;

using namespace QtCharts;

class imageMake : public QObject {
  Q_OBJECT
public:
  explicit imageMake(QObject *parent = nullptr);

signals:

public:
  void makeHistogram(QImage inimage, QCustomPlot *pchart, QImage *outimage);
  void makeHistogramAveraging(QImage inimage, QCustomPlot *pchart,
                              QImage *outimage);
  void makeBinarization(QImage inimage, QImage *outimage, quint8 threshold);
  void makeAdaptiveThreshold(QImage inimage, QImage *outimage1,
                             QImage *outimage2, quint8 area, quint8 numC);
  void makeQtsu(QImage inimage, QImage *outimage);
  void makeSobel(QImage inimage, QImage *outimage, quint8 k);

  void makeScharr(QImage inimage, QImage *outimage);
  void makeLaplace(QImage inimage, QImage *outimage, quint8 k);
  void makeCanny(QImage inimage, QImage *outimage, quint8 threshold);
  void makeHoughLines(QImage inimage, QImage *outimage, quint32 threshold);
  void makeHoughLinesP(QImage inimage, QImage *outimage, quint32 threshold);
  void makeMeanFilter(QImage inimage, QImage *outimage, quint32 k);
  void makeGaussianFilter(QImage inimage, QImage *outimage, quint32 k);
  void makeMedianFilter(QImage inimage, QImage *outimage, quint32 k);
  void makeBilateralFilter(QImage inimage, QImage *outimage, quint32 k,
                           quint32 sigmaColor, quint32 sigmaSpace);
  cv::Mat QImageToMat(QImage image);
  QImage MatToQImage(const cv::Mat &mat);
};

#endif // IMAGEMAKE_H
