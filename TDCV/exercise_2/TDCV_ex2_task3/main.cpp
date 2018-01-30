#include <stdio.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <sstream>
#include "rf.h"

using namespace cv;
using namespace std;

string IntToString (int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}


vector<float> compute_hog(Mat image1)
{
    //initialize
    Mat dst, image;
    //resize(image1,image,Size(112,112));
    resize(image1,image,Size(128,128));
    //resize(image1,image,Size(100,100));
    //resize(image1,image,Size(144,144));
    //resize(image1,image,Size(96,96));
    vector<float> descriptors;
    int height = image.rows;
    int width  = image.cols;

    //convert to gray
    cvtColor(image,dst,COLOR_RGB2GRAY);
    //hog
    //window size, block size, block stride, cell size
    //HOGDescriptor hog(Size(width-width%16, height-height%16), Size(16, 16), Size(16, 16), Size(8, 8), 9);
    HOGDescriptor hog(Size(width-width%16, height-height%16), Size(16, 16), Size(16, 16), Size(8, 8), 30);
    //HOGDescriptor hog(Size(width, height), Size(20, 20), Size(10, 10), Size(10, 10), 15);
    //HOGDescriptor hog(Size(width-width%16, height-height%16), Size(16, 16), Size(16, 16), Size(16, 16),18);
    //HOGDescriptor hog(Size(width-width%32, height-height%32), Size(32, 32), Size(16, 16), Size(32, 32),
    //9);
    resize(image1,image,Size(100,100));
    // compute hog
    // window stride, padding size
    hog.compute( dst, descriptors, Size( 8, 8 ),Size(0,0) );
    //hog.compute( dst, descriptors, Size( 5, 5 ),Size(0,0) );
    return descriptors;
}

int main(int argc, char** argv )
{

    //visualizeHOG(image,descriptors,hog,5);
    Ptr<ml::DTrees> model =  ml::DTrees::create();
    // store to be traind hog features
    Mat matrix;
    // training labels
    Mat labels;
    // loop through all classes
    for(int k=0;k<=3;k++)
    {
        // loop through directory in a class
        vector<String> filenames;
        String folder = "./train/0"+IntToString(k);
        glob(folder,filenames);
        for(int i=0; i<filenames.size();++i)
          {
            // original image
            Mat src = imread(filenames[i]);
            //  data processing
            if(!src.data)
               cerr << "Problem loading image!!!" << endl;

            cout << "current i is " << i << endl;
            vector<float> output =  compute_hog(src);
            cout << "descriptor size" << output.size() <<endl;

            //  add to features and labels
            Mat m = Mat(1,output.size(),CV_32F);
            memcpy(m.data,output.data(),output.size()*sizeof(float));
            m = m.reshape(1,1);
            matrix.push_back(m);
            labels.push_back(k);

            /*
            // flipped images
            // flip over x, horizontally, vertically

            for(int j=-1; j<=1; ++j)
            {
                Mat src_f;
                flip(src,src_f,j);
                vector<float> output =  compute_hog(src_f);
                Mat m = Mat(1,output.size(),CV_32F);
                memcpy(m.data,output.data(),output.size()*sizeof(float));
                m = m.reshape(1,1);
                matrix.push_back(m);
                labels.push_back(k);
            }

            // rotate images
            for(int j=0; j<=2; ++j)
            {
                Mat src_r;
                if(j==0) {
                    flip(src,src_r,ROTATE_90_CLOCKWISE );
                } else if (j==1){
                    flip(src,src_r,ROTATE_180);
                } else if (j==2){
                    flip(src,src_r,ROTATE_90_COUNTERCLOCKWISE );
                }
                vector<float> output =  compute_hog(src_r);
                Mat m = Mat(1,output.size(),CV_32F);
                memcpy(m.data,output.data(),output.size()*sizeof(float));
                m = m.reshape(1,1);
                matrix.push_back(m);
                labels.push_back(k);
            }
    */
          }
    }
    // convert data format for training
    labels.convertTo(labels,CV_32S);

    RandomForest1 M(32);


    M.train(matrix,labels);
    //cout<<"training finished"<<endl;
    // generate bounding box
    int image_width = 640;
    int image_height = 480;
    int slide_size = 10;
    int i=0,j=0;
    int correct_number = 0;
    int precision_number = 0;
    int recall_number = 0;

    // loop over all test images
    //
    vector<String> test_filenames;
    String test_folder = "./test";
    glob(test_folder,test_filenames);
    for(int num=0; num<test_filenames.size();++num)
    //for(int num=0; num<1;++num)
    {
        // original image
        Mat test_image = imread(test_filenames[num]);
    cout << "size of the image "<< test_image.cols << test_image.rows;


    Mat class_0, class_1, class_2;
    std::vector<cv::Rect> srcRects_0, srcRects_1, srcRects_2;
    //Mat test_windows;
    //Mat window_pos;
    // sliding window
    //for(int window_size=32;window_size<=243;window_size=window_size/2*3){
    for(int window_size=60;window_size<=140;window_size=window_size+20){
        Mat test_windows;
        Mat window_pos;
        for(int i=0; i+window_size<= image_width; i=i+slide_size){
            for(int j=0; j+window_size<= image_height; j=j+slide_size){
            //for(int j=0; j<= 0; j=j+slide_size){
                Rect window(i,j,window_size,window_size);
                //cout<<"window is " <<window << endl;
                Mat croppedImage = test_image(window);
                // add the matrix to test data
                vector<float> output =  compute_hog(croppedImage);
                Mat m = Mat(1,output.size(),CV_32F);
                memcpy(m.data,output.data(),output.size()*sizeof(float));
                m = m.reshape(1,1);
                test_windows.push_back(m);

                Vec2f pair = Vec2f(float(i),float(j));
                window_pos.push_back(pair);
            }
        }


    Mat test_output = M.predict(test_windows);
    hconcat(test_output, window_pos, test_output);
    //cout << "classification result is: " << test_output <<endl;

    // selecting potential boxes
    float thres = 0.58;
    //Mat class_0, class_1, class_2;
    //std::vector<cv::Rect> srcRects_0, srcRects_1, srcRects_2;
    for(int i=0; i<test_output.size().height; ++i){
          float a = test_output.at<float>(i,0);
          float b = test_output.at<float>(i,1);
          float c = test_output.at<float>(i,2);
          float d = test_output.at<float>(i,3);

          if(a==0 & b>=thres) {
              class_0.push_back(test_output.row(i));
              srcRects_0.push_back(cv::Rect(cv::Point(c, d), cv::Point(c+window_size, d+window_size)));
              //cout<< "a is " << a << " b is "<< b <<" row is "<< test_output.row(i) << endl;
          } else if (a==1 & b>=thres){
              class_1.push_back(test_output.row(i));
              srcRects_1.push_back(cv::Rect(cv::Point(c, d), cv::Point(c+window_size, d+window_size)));
          } else if (a==2 & b>=thres){
              class_2.push_back(test_output.row(i));
              srcRects_2.push_back(cv::Rect(cv::Point(c, d), cv::Point(c+window_size, d+window_size)));
          }
    }
     cout <<"working fine "<<endl;
    }
    cout << "matrix 0 is: " << class_0 <<endl;
    cout << "matrix 1 is: " << class_1 <<endl;
    cout << "matrix 2 is: " << class_2 <<endl;

    std::vector<cv::Rect> resRects_0,resRects_1,resRects_2;
    float a =0.05;
    nms(srcRects_0,resRects_0,a);
    nms(srcRects_1,resRects_1,a);
    nms(srcRects_2,resRects_2,a);

    //recall number
    int correct=0;
    // precision_number recall_number
    //size = resRects_0.size();
    //if(size==0)


    for(int i=0; i<resRects_0.size(); ++i){
        rectangle(test_image,resRects_0[i],Scalar(255,0,0));
        std::cout << "final rect is "<< resRects_0[i] << endl;
    }


    // processing class 1 first

    for(int i=0; i<resRects_1.size(); ++i){
        rectangle(test_image,resRects_1[i],Scalar(0,255,0));
        std::cout << "final rect is "<< resRects_1[i] << endl;
    }


    for(int i=0; i<resRects_2.size(); ++i){
        rectangle(test_image,resRects_2[i],Scalar(0,0,255));
        std::cout << "final rect is "<< resRects_2[i] << endl;
    }

    imwrite("./result/"+IntToString(num)+".jpg",test_image);


    //PR curve
    //getting ground truth rect
    ifstream File;
    File.open("./gt/00"+IntToString(num)+".gt.txt");
    int count=15;
    int numbers[count];       //allowed since C++11
    for(int a = 0; a < count; a++){
        File >> numbers[a];
        //cout<<numbers[a]<<endl;
    }
    cv::Rect gt_0 = cv::Rect(cv::Point(numbers[1], numbers[2]), cv::Point(numbers[3], numbers[4]));
    cv::Rect gt_1 = cv::Rect(cv::Point(numbers[6], numbers[7]), cv::Point(numbers[8], numbers[9]));
    cv::Rect gt_2 = cv::Rect(cv::Point(numbers[11], numbers[12]), cv::Point(numbers[13], numbers[14]));
    //cout<<gt_2<<endl;
    File.close();

    //check correcct number
    correct_number = correct_number + get_correct_number(resRects_0,gt_0) + get_correct_number(resRects_1,gt_1) + get_correct_number(resRects_2,gt_2);
    precision_number = precision_number + resRects_0.size() + resRects_1.size() + resRects_2.size();
    recall_number = recall_number + check_truth(resRects_0,gt_0) +  check_truth(resRects_1,gt_1) + check_truth(resRects_2,gt_2);
    }


    //plot the graph
    float precision = float(correct_number)/precision_number;
    cout << "precision is " << precision <<endl;
    cout << "recall is " << float(recall_number)/132 <<endl;
    //rectangle(test_image,resRects[0],Scalar(0,255,0));
    //namedWindow( "show", WINDOW_AUTOSIZE );
    //imshow("show",test_image);
    //waitKey(0);
    return 0;
}

