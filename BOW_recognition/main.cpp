#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//number of sets in database
int lib_num;
//number of images in each set
int image_number;

//test image
std::string testName = "test/b";

//testing number
int N = 1;

//matcher, extractor, and detector
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

//SIFT
cv::DescriptorExtractor * extractor = new cv::SIFT();
cv::FeatureDetector * detector = new cv::SIFT(500);

//bowtrainer
int dictionarySize;
BOWImgDescriptorExtractor bowDE(extractor, matcher);

int main()
{
    clock_t t1 = 0, t2;
    //t1 = clock();
    
    cv::Mat testImage;

    //Setting for reading the undistortFPs files
    FILE *dicData;
    char dicDataName[30] = "dictionary.txt";
    if((dicData = fopen(dicDataName, "r")) == NULL)
    {
        printf("File Not Existing...\n");
    }
    
    if(!fscanf(dicData, "Library number = %d\n", &lib_num)){
        printf("txt file format error - lib_num!!\n");
        return 0;
    };
    
    if(!fscanf(dicData, "Image number = %d\n", &image_number)){
        printf("txt file format error - img_num!!\n");
        return 0;
    };
    
    if(!fscanf(dicData, "Dictionary Size = %d\n", &dictionarySize)){
        printf("txt file format error - dictionarySize!!\n");
        return 0;
    };
    
    Mat dictionary(dictionarySize, 128, CV_32FC1);
    fscanf(dicData, "Dictionary: \n");
    for (int i = 0; i < dictionarySize; i++){
        for (int j = 0; j < 128; j++){
            fscanf(dicData, "%f ",  &dictionary.at<float>(i, j));
        }
        fscanf(dicData, "\n");
    }
    
    bowDE.setVocabulary(dictionary);
    
    //Read svm file
    CvSVM svm;
    svm.load("svm.yml");
    
    vector<KeyPoint> keypoint;
    Mat bowDescriptor;
    //inialize svm results
    int response = -1;
    
    for(int i = 0; i < N; i++){
        //load image and match in dictionary
        //testImage = cv::imread(testName + std::to_string(i) + ".jpg", 0);
        testImage = cv::imread(testName + std::to_string(12) + ".jpg", 0);
        
         //Resize the image to 900*1200
        int iImageCol = 1200;    //900
        int iImageRow = 900;     //1200
        cv::resize(testImage, testImage, Size(iImageCol, iImageRow));
        detector->detect(testImage, keypoint);
        bowDE.compute(testImage, keypoint, bowDescriptor);
        
        t1 = clock();
        //predict in svm
        response = svm.predict(bowDescriptor);
        
        //Show recognition result
        switch(response){
            case 0:
                cout << i << ". 雞肉派\n";
                break;
            case 1:
                cout << i << ". 丹麥輕乳酪\n";
                break;
            case 2:
                cout << i << ". 丹麥田園\n";
                break;
            case 3:
                cout << i << ". 丹麥蜂蜜年輪\n";
                break;
            case 4:
                cout << i << ". 菠蘿起士包\n";
                break;
        }
        
        //if (i == 7){
            for (int x = 0; x < bowDescriptor.cols; x++){
                cout << (int) bowDescriptor.at<uchar>(x, 0) << endl;
            }
        //}
    }
    
    fclose(dicData);
    
    t2 = clock();
    float time_diff((float)t2 - (float)t1);
    float sec = time_diff/CLOCKS_PER_SEC;
    printf("running time = %f\n", sec);
    
    return 0;
}
