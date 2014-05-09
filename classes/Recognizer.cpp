//
//  Recognizer.cpp
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#include "Recognizer.h"

Recognizer::Recognizer(float nnndrRatio, float tthresh) {
    books.clear();
    nndrRatio = nnndrRatio;
    thresh = tthresh;
}

void Recognizer::train(uint nid) {
    books.push_back(*new Book(nid));
}

void Recognizer::train(uint nid, Mat dataSet) {
    Mat descriptors;
    vector<KeyPoint> keypoints;
    detector.detect(dataSet, keypoints);
    extractor.compute(dataSet, keypoints, descriptors);
    books.push_back(*new Book(nid, keypoints, descriptors));
}

void Recognizer::storeTrainingSet(uint nid, Mat dataSet) {
    Mat descriptors;
    vector<KeyPoint> keypoints;
    SurfFeatureDetector detector;
    detector.detect(dataSet, keypoints);
    SurfDescriptorExtractor extractor;
    extractor.compute(dataSet, keypoints, descriptors);
    Book *b = new Book(nid, keypoints, descriptors);
    cout << "Storing: " << b << endl;
    b->SaveData();
    dataSet.release();
}

bool Recognizer::recognize(Mat warpedCover) {
    currentCover.release();
    currentCover = warpedCover.clone();
    detector.detect(warpedCover, currentKeypoints);
    if(currentKeypoints.size() == 0){
        return false;
    }
    extractor.compute(warpedCover, currentKeypoints, currentDescriptors);
    vector<double> confidence;
    DMatch m1, m2;
    int maxIndex = 0;
    long maxCount = 0;
    for(int i = 0; i < books.size(); ++i) {
        matches.clear();
        matcher.knnMatch(books.at(i).Descriptors(), currentDescriptors, matches, 2);
        
        good_matches.clear();
        
        for (int j = 0; j < matches.size(); ++j)
        {
            if (matches[j].size() < 2) {
                continue;
            }
            
            m1 = matches[j][0];
            m2 = matches[j][1];
            
            if(m1.distance <= nndrRatio * m2.distance) {
                good_matches.push_back(m1);
            }
        }
                
        if(good_matches.size() > 7 && good_matches.size() > maxCount)
        {
            maxCount = good_matches.size();
            maxIndex = i;
        }
        
        cout << books.at(i).Id() << "; matches count: " << good_matches.size() << "\n";
        
        confidence.push_back((double)good_matches.size() / matches.size());
        
        cout << "Confidence: " << confidence.at(i) * 100 << "% \n";
    }
    
    if(maxCount == 0){
        return false;
    } else {
        lastResult = books.at(maxIndex);
        cout << ((confidence.at(maxIndex) > thresh) ? "Good match." : "Propably bad.") << "\n#CONFIDENCE: " << confidence.at(maxIndex) << "\n";
        return (confidence.at(maxIndex) > thresh);
    }
}

Book Recognizer::LastResult() {
    return lastResult;
}