//
//  main.cpp
//  arbc
//
//  Created by Árpád Kiss on 2014.03.06..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "classes/Preprocessor.h"
#include "classes/Recognizer.h"
#include <iomanip>
#include "classes/Book.h"
#include <ctime>

using namespace cv;
using namespace std;

void recognizing(string filename, vector<uint> ids) {

	Mat image = imread(filename);

	Recognizer* rcg = new Recognizer(0.70f, 0.065f);
	Preprocessor* pr = new Preprocessor(image);
    pr->preprocess(10);

    Mat cover;

    if(!pr->warpObject()) {
    	cout << "#WARP-UNSUCCESSFUL" << endl;
    	cover = image.clone();
    	image.release();
    	Size newSize = (cover.size().width > cover.size().height) ? Size(1024, 768) : Size(768, 1024);
    	resize(cover, cover, newSize);
    	cout << "using the whole image..." << endl;
    } else {
    	cover = pr->WarpedCover();
    }

    Mat trainingImage;

    for(size_t i = 0; i < ids.size(); ++i) {
    	cout << "training " << i << endl;
    	rcg->train(ids.at(i));
	}

	if(rcg->recognize(Preprocessor::kuwaharaNagaoFilter(cover)))
	{
   		Book result = rcg->LastResult();
        cout << "#BEGIN-RESULT " << result.Id() << " #END-RESULT";
    } else {
    	cout << "#BEGIN-RESULT null #END-RESULT";
    }
    cout << endl;
}

void dummyTrain() {
	vector<string> trainingData;
	trainingData.push_back("data-set/beautiful_teams.jpg");
	trainingData.push_back("data-set/good_parts.jpg");
	trainingData.push_back("data-set/hvg_b_forradalmarok.jpg");
	trainingData.push_back("data-set/hvg_b_velemenyvezerek.jpg");
	trainingData.push_back("data-set/smashing_1.jpg");
	trainingData.push_back("data-set/being_geek.jpg");
	trainingData.push_back("data-set/high_perf_web_sites.jpg");
	trainingData.push_back("data-set/hvg_b_kreatorok.jpg");
	trainingData.push_back("data-set/js_web_apps.jpg");
	trainingData.push_back("data-set/smashing_2.jpg");

	Mat image;

	for(size_t i = 0; i < trainingData.size(); ++i)
    { 
    	image = imread(trainingData.at(i));
    	Recognizer::storeTrainingSet((uint)(i + 1), Preprocessor::kuwaharaNagaoFilter(image));
    	cout << "#BEGIN-DATA-SET: " + to_string((uint)(i + 1)) << endl;
    	cout << "#BEGIN-CLUSTERVECTOR" << endl;
		cout << Preprocessor::calculateLAB(40, image.clone()) << endl;
		cout << "#END-CLUSTERVECTOR" << endl;
		cout << "#END-DATA-SET" << endl;
    }
}

void storing(uint id, string file) {
	Mat image = imread(file);
	Recognizer::storeTrainingSet(id, Preprocessor::kuwaharaNagaoFilter(image));
	cout << "#BEGIN-DATA-SET: " + to_string(id) << endl;
	cout << "#BEGIN-CLUSTERVECTOR" << endl;
	cout << Preprocessor::calculateLAB(40, image.clone()) << endl;
	cout << "#END-CLUSTERVECTOR" << endl;
	cout << "#END-DATA-SET" << endl;
}

string prependStringToFileName(string p, string s) {

   char sep = '/';

   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      return (s.substr(0, i+1)) + p + (s.substr(i+1, s.length() - i));
   }

   return("");
}

void clustering(string filename) {
	Mat image = imread(filename);

	Preprocessor* pr = new Preprocessor(image);
    pr->preprocess(10);

    Mat cover;

    if(!pr->warpObject()) {
    	cout << "#WARP-UNSUCCESSFUL" << endl;
    	cover = image.clone();
    	image.release();
    	Size newSize = (cover.size().width > cover.size().height) ? Size(1024, 768) : Size(768, 1024);
    	resize(cover, cover, newSize);
    	cout << "using the whole image..." << endl;
    } else {
    	cover = pr->WarpedCover();
    	string nf = prependStringToFileName("to_swt_", filename);
    	//Mat changeable;
    	//cover.convertTo(changeable, 1, 30);
    	imwrite(nf, cover);
    	cout << "#BEGIN-TO-SWT" << endl;
    	cout << nf << endl;
    	cout << "#END-TO-SWT" << endl;
    }

	cout << "#BEGIN-CLUSTERVECTOR" << endl;
	cout << Preprocessor::calculateLAB(40, cover) << endl;
	cout << "#END-CLUSTERVECTOR" << endl;
}

void ocr(string swtFilename) {
	Mat image = imread(swtFilename);
	cout << "#BEGIN-KEYWORDS" << endl;
	cout << Preprocessor::processOCR(image);
	cout << "#END-KEYWORDS" << endl;
}

int main( int argc, char** argv )
{
	clock_t begin_time = clock();

	if(argc < 2) {
		cout << "error: invalid arguments" << endl;
		exit(1);
	}

	string mode(argv[1]);

	if( mode == "-cluster" ) {
		if(argc < 3) {
			cout << "error: invalid arguments" << endl;
			exit(1);
		}
		string recognizable(argv[2]);
		cout << "Clustering: " << recognizable << endl;
		clustering(recognizable);
	} else if ( mode == "-recognize" ) {
		if(argc < 3) {
			cout << "error: invalid arguments" << endl;
			exit(1);
		}
		string recognizable(argv[2]);

		cout << "Recognizing: " << recognizable << endl;

		vector<uint> ids;

		for(uint i = 3; i < argc; i++) {
			ids.push_back(atoll(argv[i]));
		}

		recognizing(recognizable, ids);
	} else if ( mode == "-store" ) {
		string storeable(argv[2]);

		if(argc < 4) {
			cout << "error: invalid arguments" << endl;
			exit(1);
		}

		uint id = atoll(argv[3]);

		cout << "Storing: " << "ID: " + to_string(id) + " :: " << storeable << endl;
		storing(id, storeable);
	} else if ( mode == "-ocr" ) { 
		string processable(argv[2]);
		ocr(processable);
	} else if ( mode == "-dummy-train" ) { 
		dummyTrain();
	} else if ( mode == "--help" || mode == "-h" ) { 
		cout << "Usage:" << endl;
		cout << " -ocr filepath" << endl;
		cout << " -recognize filepath [posible-book-id, ...]" << endl;
		cout << " -store filepath new-id" << endl;
		cout << " -cluster filepath" << endl;
	} else {
		cout << "Error: mode not found." << endl;
		exit(1);
	}

	cout << "Execution took: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << "ms" << endl;

	return 0;
}