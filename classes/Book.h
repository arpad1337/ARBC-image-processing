//
//  Book.h
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#ifndef __arbc__Book__
#define __arbc__Book__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <unistd.h>

using namespace cv;
using namespace std;

class Book {
public:
    Book();
    Book(uint nid, vector<KeyPoint> nkeypoints, Mat ndescriptors);
    Book(uint nid);
    Mat Descriptors();
    vector<KeyPoint> Keypoints();
    uint Id();
    void SaveData();
    void LoadData();
private:
    vector<KeyPoint> keypoints;
    Mat descriptors;
    uint id;
};

#endif /* defined(__arbc__Book__) */
