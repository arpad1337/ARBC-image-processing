//
//  Preprocessor.h
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#ifndef __arbc__Preprocessor__
#define __arbc__Preprocessor__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;
using namespace tesseract;

class Preprocessor {
public:
    Preprocessor(Mat image);
    void preprocess(int nthreshold);
    Mat Result();
    Mat ResultOnOriginal();
    Mat WarpedCover();
    bool warpObject();
    static Mat kuwaharaNagaoFilter(Mat wc);
    static vector<Point2d> calculateLAB(uint padding, Mat image);
    static string processOCR(Mat imageMat);
private:
    int thresh;
    Mat preprocessable;
    Mat preprocessableGrey;
    Mat warpedCover;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    void SNNEdgeDetection();
    void HoughLines();
    Size newSize;
    double getOrientation(vector<Point> &pts, Mat &img);
};

#endif /* defined(__arbc__Preprocessor__) */
