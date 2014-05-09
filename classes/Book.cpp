//
//  Book.cpp
//  arbc
//
//  Created by Árpád Kiss on 2014.03.10..
//  Copyright (c) 2014 rpi's lab. All rights reserved.
//

#include "Book.h"

Book::Book(uint nid, vector<KeyPoint> nkeypoints, Mat ndescriptors) {
    id = nid;
    keypoints = nkeypoints;
    descriptors = ndescriptors;
}

Book::Book(uint nid) {
    id = nid;
    LoadData();
}

Book::Book() {
    
}

Mat Book::Descriptors() {
    return descriptors;
}

vector<KeyPoint> Book::Keypoints(){
    return keypoints;
}

uint Book::Id(){
	return id;
}

void Book::SaveData() {
    FileStorage fs("storage/" + to_string(id) + ".xml", FileStorage::WRITE);
    fs << "keypoints" << keypoints;
    fs << "descriptors" << descriptors;
    fs.release();
}
void Book::LoadData() {
    FileStorage fs("storage/" + to_string(id) + ".xml", FileStorage::READ);
    read(fs["keypoints"], keypoints);
    read(fs["descriptors"], descriptors);
    fs.release();
}
