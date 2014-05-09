//
//  Preprocessor.cpp
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#include "Preprocessor.h"

Preprocessor::Preprocessor(Mat image) {
    preprocessable.release();
    preprocessableGrey.release();
    preprocessable = image.clone();
    preprocessableGrey = Mat(Mat::zeros(preprocessable.size().height, preprocessable.size().width, CV_8UC1));
    cvtColor(preprocessable, preprocessableGrey, CV_BGR2GRAY);
    warpedCover = Mat(Mat::zeros(500,380,CV_8UC3));
    newSize = (preprocessable.size().width > preprocessable.size().height) ? Size(1024, 768) : Size(768, 1024);
    thresh = 10;
}

vector<Point2d> Preprocessor::calculateLAB(uint padding, Mat image) {
    
    resize(image, image, Size(380, 500));
    
    vector<Point2d> meanIntensities;
    
    uint hStep = (image.size().width - (padding << 1)) >> 1;
    uint vStep = (image.size().height - (padding << 1)) / 3;
    
    Mat block;
    Scalar meanIntensity, stdDev;
    
    for(uint i = 0; i < 2; i++) {
        for(uint j = 0; j < 3; j++) {
            block = image(Rect(padding + i * hStep, padding + j * vStep, hStep, vStep)).clone();
            cvtColor(block, block, CV_BGR2Lab);
            meanStdDev(block, meanIntensity, stdDev);
            meanIntensities.push_back( Point2d(meanIntensity[1], meanIntensity[0]) );
        }
    }

    return meanIntensities;
}

string Preprocessor::processOCR(Mat imageMat) {
    string result;

    TessBaseAPI tess;
    
    GenericVector<STRING> pars_vec;
    pars_vec.push_back("load_system_dawg");
    pars_vec.push_back("load_freq_dawg");
    pars_vec.push_back("user_words_suffix");
    pars_vec.push_back("tessedit_char_whitelist");
    
    GenericVector<STRING> pars_values;
    pars_values.push_back("T");
    pars_values.push_back("T");
    pars_values.push_back("user-words");
    pars_values.push_back("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'");

    tess.Init("", "eng", OEM_DEFAULT, NULL, 0, &pars_vec, &pars_values, false);

    tess.SetImage((uchar*)imageMat.data, imageMat.size().width, imageMat.size().height, imageMat.channels(), imageMat.step1());
    tess.Recognize(0);
    
    result = tess.GetUTF8Text();

    //Mat wtf = Mat(imageMat.size().height, imageMat.size().width, imageMat.type(), Scalar(255,255,255));

    //subtract(wtf,imageMat,imageMat);

    Mat fg;
    imageMat.convertTo(fg, CV_32F);
    fg = fg + 1;
    log(fg, fg);
    normalize(fg,fg,0,255,NORM_MINMAX);
    convertScaleAbs(fg,fg);

    tess.Init("", "eng", OEM_DEFAULT, NULL, 0, &pars_vec, &pars_values, false);

    tess.SetImage((uchar*)fg.data, fg.size().width, fg.size().height, fg.channels(), fg.step1());
    tess.Recognize(0);

    return result + tess.GetUTF8Text();
}


void Preprocessor::preprocess(int nthreshold) {
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1,1) );
    thresh = nthreshold;
    resize(preprocessableGrey, preprocessableGrey, newSize);
    resize(preprocessable, preprocessable, newSize);
    
    Mat center;
    Scalar meanIntensity, stdDev;
    center = preprocessable(Rect(preprocessable.size().width / 2 - 100,preprocessable.size().height / 2 - 100,200,200)).clone();
    meanStdDev(center, meanIntensity, stdDev);
        
    medianBlur(preprocessableGrey, preprocessableGrey, 3);
    
    int devth = 55;
    
    if((stdDev[0] + stdDev[1] + stdDev[2]) / 3 > devth )
    {   
        vector<Mat> channels(3);
        split(preprocessable, channels);
        int channelIndex = 0;
        for(int i = 0; i < 3; i++)
        {
            if(meanIntensity[i] < meanIntensity[channelIndex]) {
                channelIndex = i;
            }
        }
                
        thresh = 6 * thresh;
        
        preprocessableGrey = channels[channelIndex].clone();

        medianBlur(preprocessableGrey, preprocessableGrey, 5);
        
        threshold(preprocessableGrey,preprocessableGrey, meanIntensity[channelIndex], 255, THRESH_BINARY);
    } else {
        threshold(preprocessableGrey,preprocessableGrey, 50, 255, THRESH_OTSU); 
    }
    
    SNNEdgeDetection();
    
    morphologyEx(preprocessableGrey, preprocessableGrey, MORPH_CLOSE, element);
    
    HoughLines();
}

void Preprocessor::HoughLines() {
    vector<Vec4i> lines;
    HoughLinesP(preprocessableGrey, lines, 1, CV_PI/180, 60, 180, 25 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( preprocessableGrey, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), 255, 2, CV_8UC1);
    }
}

bool Preprocessor::warpObject() {
    bool warpSuccessful = false;
    int minDistance = 100;
    
    findContours( preprocessableGrey, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    vector<Point> approxCurve;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
    {
        if ( 80< contours[idx].size() && contours[idx].size()<1000  )
        {
            approxPolyDP( contours[idx], approxCurve, double ( contours[idx].size() ) * 0.2 , true);
            
            if ( approxCurve.size() ==4 && isContourConvex ( Mat(approxCurve)) )
            {
                Point2f p[4];
                Point2f q[4];
                
                p[0].x= approxCurve[0].x;
                p[0].y= approxCurve[0].y;
                p[1].x= approxCurve[1].x;
                p[1].y= approxCurve[1].y;
                
                p[2].x= approxCurve[2].x;
                p[2].y= approxCurve[2].y;
                p[3].x= approxCurve[3].x;
                p[3].y= approxCurve[3].y;
                
                if( sqrt((approxCurve[0].x-approxCurve[1].x)*(approxCurve[0].x-approxCurve[1].x)
                         + (approxCurve[0].y-approxCurve[1].y)*(approxCurve[0].y-approxCurve[1].y)) > minDistance &&
                   sqrt((approxCurve[1].x-approxCurve[2].x)*(approxCurve[1].x-approxCurve[2].x)
                        + (approxCurve[1].y-approxCurve[2].y)*(approxCurve[1].y-approxCurve[2].y)) > minDistance &&
                   sqrt((approxCurve[2].x-approxCurve[3].x)*(approxCurve[2].x-approxCurve[3].x)
                        + (approxCurve[2].y-approxCurve[3].y)*(approxCurve[2].y-approxCurve[3].y)) > minDistance &&
                   sqrt((approxCurve[3].x-approxCurve[0].x)*(approxCurve[3].x-approxCurve[0].x)
                        + (approxCurve[3].y-approxCurve[0].y)*(approxCurve[3].y-approxCurve[0].y)) > minDistance
                   )
                {
                    q[0].x= (float) 0;
                    q[0].y= (float) 0;
                    q[1].x= (float) 0;
                    q[1].y= (float) 500;
                    
                    q[2].x= (float)380;
                    q[2].y= (float)500;
                    q[3].x= (float)380;
                    q[3].y= (float)0;
                    
                    // to do: fix orientation
                    
                    p[0].x= approxCurve[0].x;
                    p[0].y= approxCurve[0].y;
                    p[1].x= approxCurve[1].x;
                    p[1].y= approxCurve[1].y;
                    
                    p[2].x= approxCurve[2].x;
                    p[2].y= approxCurve[2].y;
                    p[3].x= approxCurve[3].x;
                    p[3].y= approxCurve[3].y;
                    
                    double orientation = getOrientation(contours[idx], preprocessable);
                    
                    if( ( orientation > 0 && orientation < CV_PI / 6.5 )  ||
                       (orientation > (CV_PI / 2 - CV_PI / 13) && orientation < (CV_PI / 2 + CV_PI / 13)) ||
                        (orientation > (CV_PI - CV_PI / 6.5) && orientation < CV_PI ))
                    {
                        p[0].x= approxCurve[0].x;
                        p[0].y= approxCurve[0].y;
                        p[1].x= approxCurve[1].x;
                        p[1].y= approxCurve[1].y;
                        
                        p[2].x= approxCurve[2].x;
                        p[2].y= approxCurve[2].y;
                        p[3].x= approxCurve[3].x;
                        p[3].y= approxCurve[3].y;
                    } else {
                        p[0].x= approxCurve[1].x;
                        p[0].y= approxCurve[1].y;
                        p[1].x= approxCurve[2].x;
                        p[1].y= approxCurve[2].y;
                        
                        p[2].x= approxCurve[3].x;
                        p[2].y= approxCurve[3].y;
                        p[3].x= approxCurve[0].x;
                        p[3].y= approxCurve[0].y;
                    }
                    
                    Mat warpMatrix = getPerspectiveTransform(p,q);
                    
                    warpPerspective(preprocessable, warpedCover, warpMatrix, Size(380,500),INTER_NEAREST);
                    
                    warpSuccessful = true;
                }
            }
        }
    }
    return warpSuccessful;
}

/*
 * Symmetric Nearest Neighbour
 */
void Preprocessor::SNNEdgeDetection() {
    Mat temp = preprocessableGrey.clone();
    uint z = 0;
    uchar *ptr = (uchar*)(preprocessableGrey.data);
    uchar *ptr_2 = (uchar*)(temp.data);
    uint stride = preprocessableGrey.size().width;
    uint i = stride * 3 + 1;
    uint length = (preprocessableGrey.size().width - 1 )*(preprocessableGrey.size().height - 1);
    int variances[4];
    int _max;
    for(; i < length; i++)
    {
        _max = 0;
        variances[0] = abs(ptr[i-1 -stride] - ptr[i+1 + stride]);
        variances[1] = abs(ptr[i - stride] - ptr[i + stride]);
        variances[2] = abs(ptr[i+1 - stride] - ptr[i - 1 + stride]);
        variances[3] = abs(ptr[i - 1] - ptr[i + 1]);
            
        for(z = 0; z<4; z++)
        {
            if(variances[z] > variances[_max])
            {
                _max = z;
            }
        }
        ptr_2[i] = (variances[_max] > thresh)?255:0;
    }
    preprocessableGrey = temp.clone();
    temp.release();
}


Mat Preprocessor::kuwaharaNagaoFilter(Mat wc)
{
    int kuwaharaLut_2[1024];
    int kw = 0;
    for(int i = 0; i< 1024 ; i+=4)
    {
        kuwaharaLut_2[i] = kw;
        kuwaharaLut_2[i+1] = kw;
        kuwaharaLut_2[i+2] = kw;
        kuwaharaLut_2[i+3] = kw;
        kw++;
    }
    
    int stride, height, i, z, j, _min, k;
    
    double means[4][3];
    double var[4];
    
    Mat currentFrame;
    
    uchar *ptr_2;
    vector<Mat> channels(3);
    currentFrame = wc.clone();
    ptr_2 = (uchar*)(currentFrame.data);
    double windowMean = 0;
    stride = wc.size().width;
    height = wc.size().height;

    for(i=1;i<height-2;++i){
        for(j=1;j<stride-2;++j) {
            var[0] = 0.0;
            var[1] = 0.0;
            var[2] = 0.0;
            var[3] = 0.0;

            for(z=0;z<3;z++) // by color channels
            {
                 
                means[0][z] = kuwaharaLut_2[wc.at<Vec3b>(i-1,j-1)[z]
                                            + wc.at<Vec3b>(i,j-1)[z]
                                            + wc.at<Vec3b>(i-1,j)[z]
                                            + wc.at<Vec3b>(i,j)[z]];
                means[1][z] = kuwaharaLut_2[wc.at<Vec3b>(i-1,j)[z]
                                            + wc.at<Vec3b>(i,j)[z]
                                            + wc.at<Vec3b>(i-1,j+1)[z]
                                            + wc.at<Vec3b>(i,j+1)[z]];
                means[2][z] = kuwaharaLut_2[wc.at<Vec3b>(i,j-1)[z]
                                            + wc.at<Vec3b>(i+1,j-1)[z]
                                            + wc.at<Vec3b>(i,j)[z]
                                            + wc.at<Vec3b>(i,j+1)[z]];
                means[3][z] = kuwaharaLut_2[wc.at<Vec3b>(i,j)[z]
                                            + wc.at<Vec3b>(i,j+1)[z]
                                            + wc.at<Vec3b>(i,j+1)[z]
                                            + wc.at<Vec3b>(i+1,j+1)[z]];
                
                windowMean = 0;
                for(int k = -1; k < 2; k++)
                {
                    for(int l = -1; l < 2;l++) {
                        windowMean += wc.at<Vec3b>(i - k,j - l)[z];
                    }
                }
                
                windowMean = windowMean / 9;
                
                var[0] += (means[0][z] - windowMean) * (means[0][z] - windowMean);
                var[1] += (means[1][z] - windowMean) * (means[1][z] - windowMean);
                var[2] += (means[2][z] - windowMean) * (means[2][z] - windowMean);
                var[3] += (means[3][z] - windowMean) * (means[3][z] - windowMean);

            }
        
            _min = 10000;
            k=0;
            for(z=0;z<4;z++)
            {
                if(var[z] < _min)
                {
                    _min = var[z];
                    k = z;
                }
            }
            
            ptr_2[(i*stride+j)*3] = means[k][0];
            ptr_2[(i*stride+j)*3+1] = means[k][1];
            ptr_2[(i*stride+j)*3+2] = means[k][2];   

        }
    }

    return currentFrame;
}

/*
 PCA
*/

double Preprocessor::getOrientation(vector<Point> &pts, Mat &img)
{
    Mat data_pts = Mat((int)pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    
    Point pos = Point(pca_analysis.mean.at<double>(0, 0),
                      pca_analysis.mean.at<double>(0, 1));
    
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
 
    return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

Mat Preprocessor::Result() {
    return preprocessableGrey;
}

Mat Preprocessor::ResultOnOriginal() {
    return preprocessable;
}

Mat Preprocessor::WarpedCover() {
    return warpedCover;
}