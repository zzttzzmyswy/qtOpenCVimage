QT       += core gui

QT += charts
QT += printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# User start
INCLUDEPATH += /usr/include/opencv4
LIBS += /usr/lib/x86_64-linux-gnu/libopencv_*.so

# User end

SOURCES += \
    imagemake.cpp \
    main.cpp \
    qcustomplot/qcustomplot.cpp \
    widget.cpp

HEADERS += \
    imagemake.h \
    qcustomplot/qcustomplot.h \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    README.MD
