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
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

/*initialize global variables*/

//number of sets in database
int lib_num = 5;
//number of images in each set
int img_num = 60;

float progress = 0;

//matcher, extractor, and detector
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

//SIFT
cv::DescriptorExtractor * extractor = new cv::SIFT();
cv::FeatureDetector * detector = new cv::SIFT(500);

//bowtrainer
int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

//class to extract features
void ClassExtractFeatures()
{
    cv::Mat img;
    int i, j;
    for(j = 0; j < lib_num; j++)
    {
        for(i = 0; i<img_num; i++){
            
            std::ostringstream convert;
            convert << j << "/"  << i << ".jpg";
            img = cvLoadImage(convert.str().c_str(),0);
            
            //Resize the image to 900*1200
            int iImageCol = 1200;   //900
            int iImageRow = 900;    //1200
            cv::resize(img, img, Size(iImageCol, iImageRow));
            
            vector<KeyPoint> keypoint;
            detector->detect(img, keypoint);
            Mat features;
            extractor->compute(img, keypoint, features);
            bowTrainer.add(features);
            progress = (i + j*img_num)*100/(lib_num*img_num*2);
            cout << "progress = " << progress<< "%" << endl;
        }
    }
    return;
}

int main()
{
    clock_t t1, t2;
    t1 = clock();
    
    int i, j;
    cv::Mat img;
    
    //collect extract features
    ClassExtractFeatures();
    
    //get all descriptors
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    
    int count = 0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    cout<<"Clustering "<<count<<" features"<<endl;
    
    
    //choosing cluster's centroids as dictionary's words
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);
    
    //initialize labels and training data
    Mat labels(0, 1, CV_32FC1);
    Mat trainingData(0, dictionarySize, CV_32FC1);
    
    //initialize keypoint and bowdescriptor
    vector<KeyPoint> keypoint1;
    Mat bowDescriptor1;
    
    //extracting histogram in the form of bow for each image
    for(j = 0; j < lib_num; j++)
    {
        for(i = 0; i < img_num; i++){
            
            std::ostringstream convert;
            convert << j << "/"  << i << ".jpg";
            //load image, detect features
            img = cvLoadImage(convert.str().c_str(),0);
            //Resize the image to 900*1200
            int iImageCol = 1200;   //900
            int iImageRow = 900;    //1200
            cv::resize(img, img, Size(iImageCol, iImageRow));
            detector->detect(img, keypoint1);
            //stack in features in bow dictionary and training data
            bowDE.compute(img, keypoint1, bowDescriptor1);
            trainingData.push_back(bowDescriptor1);
            labels.push_back((float) j);
            progress = 50 + ((i + j*img_num)*100)/(lib_num*img_num*2);
            cout << "progress = " << progress << "%" << endl;
        }
    }

    //Setting for writing the training data, label, data number, image number & dictionarySize into the files
    FILE *dicData;
    char dictDataName[30] = "output/dictionary.txt";
    if((dicData = fopen(dictDataName, "w")) == NULL)
    {
        printf("File Not Existing...\n");
    }
    fprintf(dicData, "Library number = %d\n", lib_num);
    fprintf(dicData, "Image number = %d\n", img_num);
    fprintf(dicData, "Dictionary Size = %d\n", dictionarySize);
    fprintf(dicData, "Dictionary: \n");
    for (int i = 0; i < dictionary.rows; i++){
        for(int j = 0; j < dictionary.cols; j++){
            fprintf(dicData, "%lf ", dictionary.at<float>(i, j));
        }
        fprintf(dicData, "\n");
    }
    
    //Setting up SVM parameters
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.gamma = 1;
    params.p = 5e-1;
    params.C = 0.01;
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
    CvSVM svm;
    
    //save svm file with auto-adjust parameters
    CvParamGrid nuGrid = CvParamGrid(1, 1, 0.0);
    CvParamGrid coeffGrid = CvParamGrid(1, 1, 0.0);
    CvParamGrid degreeGrid = CvParamGrid(1, 1, 0.0);
    CvSVM regressor;
    regressor.train_auto(trainingData, labels, Mat(), Mat(), params, 10, regressor.get_default_grid(CvSVM::C), regressor.get_default_grid(CvSVM::GAMMA), regressor.get_default_grid(CvSVM::P), nuGrid, coeffGrid, degreeGrid);
    regressor.save("output/svm.yml");
    CvSVMParams params_re = regressor.get_params();
    float paraC = params_re.C;
    float paraP = params_re.p;
    float paraGamma = params_re.gamma;
    printf("Parms: C = %f, P = %f,gamma = %f \n",paraC,paraP,paraGamma);

    //Get running time
    t2 = clock();
    float time_diff((float)t2 - (float)t1);
    float sec = time_diff/CLOCKS_PER_SEC;
    printf("running time = %f\n", sec);
    
    return 0;
}
