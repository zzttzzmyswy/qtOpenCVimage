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
  void makeAdaptiveMedianFilter(QImage inimage, QImage *outimage);
  uchar adaptiveProcess(const Mat &im, int row, int col, int kernelSize,
                        int maxSize);
  void makeBilateralFilter(QImage inimage, QImage *outimage, quint32 k,
                           quint32 sigmaColor, quint32 sigmaSpace);
  void makeFrequencyDfilter(QImage inimage, QImage *outimage1,
                            QImage *outimage2, quint32 dD, quint32 nN,
                            quint8 type);
  void makeUSM(QImage inimage, QImage *outimage, float w, uint32_t Threshold);
  void makeLaplacianSharpen(QImage inimage, QImage *outimage);
  void makeHomomorphicFilter(QImage inimage, QImage *outimage, double D0,
                             double gammaH, double gammaL, double c);
  void makeRotate(QImage inimage, QImage *outimage1, QImage *outimage2,
                  QImage *outimage3);

  void makeOutOfFocusDeblurFilter(QImage inimage, QImage *outimage, quint64 r,
                                  quint64 snr);
  void makeUniformNoise(QImage inimage, QImage *outimage, quint64 a1,
                        quint64 a2);
  void makeGaussianNoise(QImage inimage, QImage *outimage, quint64 mean,
                         quint64 sd);
  void makeSaltPepperNoise(QImage inimage, QImage *outimage, quint64 t);
  cv::Mat QImageToMat(QImage image);
  QImage MatToQImage(const cv::Mat &mat);
  cv::Mat openCvFreqFilt(cv::Mat scr, cv::Mat blur);
  void fftshift(const Mat &inputImg, Mat &outputImg);
  void edgetaper(const Mat &inputImg, Mat &outputImg, double gamma = 5.0,
                 double beta = 0.2);
  void filter2DFreq(const Mat &inputImg, Mat &outputImg, const Mat &H);
};
#endif // IMAGEMAKE_H
