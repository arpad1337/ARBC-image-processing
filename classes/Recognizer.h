//
//  Recognizer.h
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#ifndef __arbc__Recognizer__
#define __arbc__Recognizer__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "Book.h"

using namespace cv;
using namespace std;

class Recognizer {
public:
    Recognizer(float nnndrRatio, float tthresh);
    void train(uint nid, Mat dataSet);
    void train(uint nid);
    bool recognize(Mat preprocessedImage);
    Book LastResult();
    static void storeTrainingSet(uint nid, Mat dataSet);
private:
    float nndrRatio;
    float thresh;
    SurfFeatureDetector detector;
    SurfDescriptorExtractor extractor;
    FlannBasedMatcher matcher;
    Mat currentCover;
    vector<KeyPoint> currentKeypoints;
    Mat currentDescriptors;
    vector<Book> books;
    vector< vector< DMatch >  > matches;
    vector< DMatch > good_matches;
    Book lastResult;
};

#endif /* defined(__arbc__Recognizer__) */
