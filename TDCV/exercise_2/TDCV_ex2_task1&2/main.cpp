//
// Created by highstars1 on 31.12.17.
//

#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/ml/ml.hpp>
#include <string>
#include "hog_visualization.hpp"


using namespace std;
using namespace cv;
using namespace cv::ml;

enum RotateFlags {
    ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
    ROTATE_180 = 1, //Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
};

// visualize the HOG descriptor
void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor);

// GetRandomInteger() and randperm are used to generate random index
unsigned int GetRandomInteger(int low, int up)
{
    unsigned int uiResult;

    if (low > up)
    {
        int temp = low;
        low = up;
        up = temp;
    }

    uiResult = static_cast<unsigned int>((rand() % (up - low + 1)) + low);

    return uiResult;
}

int* randperm(int Num, int low, int up) {

    auto *temp = new int[Num];
    for(int i = 0; i < Num; ++i){
        temp[i]=GetRandomInteger(low,up);
     }
    return temp;
 }


// 输入是vote_all_trees的每一行,每一行是一张图片100棵树的分类结果
int vote_for_result(const float a[], int num_of_trees){
    float vote[num_of_trees];
    vote[num_of_trees] = {0};
    for(int i=0;i<num_of_trees;i++){
        for(int j=0;j<num_of_trees;j++){
            if(a[i]==a[j]){
                vote[i] = vote[i] + 1;
            }
        }
    }

    //两两比较后输出最大的那个
    float vote_max = 0;
    int vote_max_index=0;
    for(int i=0;i<num_of_trees;i++){
        if(vote[i]>vote_max){
            vote_max = vote[i];
            vote_max_index = i;
        }
    }
    return vote_max_index;
}



class randomForest{
public:

    randomForest(){
        cv::Ptr<cv::ml::DTrees> dtrees;
        dtrees = cv::ml::DTrees::create();
        dtrees->setMaxDepth(10);
        dtrees->setMinSampleCount(10);
        dtrees->setMaxCategories(2);
        dtrees->setCVFolds(0);
    }

    cv::Ptr<cv::ml::DTrees> creatRF(){
        cv::Ptr<cv::ml::DTrees> dtrees;
        dtrees = cv::ml::DTrees::create();
        dtrees->setMaxDepth(10);
        dtrees->setMinSampleCount(10);
        dtrees->setMaxCategories(2);
        dtrees->setCVFolds(0);
        //rftree = dtrees;
        return dtrees;
    }

    void trainRF(const cv::Ptr<cv::ml::DTrees> &tree, const cv::Ptr<cv::ml::TrainData> &data){
        tree->train(data);
    }


};


int main() {

    /*
    // task 1 ///////////////image processing and HOG descriptor///////////////

    // load original image
    cv::Mat image;
    image = cv::imread("/users/gaoyingqiang/Desktop/data/task1/obj1000.jpg");
    //cv::namedWindow("obj1000", CV_WINDOW_NORMAL);
    //cv::imshow("obj1000",image);

    // convert to gray image
    cv::Mat gray_image;
    cv::cvtColor(image,gray_image,CV_RGB2GRAY);
    cv::imwrite("/users/gaoyingqiang/Desktop/data/task1/gray_image.jpg", gray_image);

    // flip the image
    cv::Mat flip_image;
    cv::flip(image, flip_image,1);
    cv::imwrite("/users/gaoyingqiang/Desktop/data/task1/flip_image.jpg", flip_image);

    // rotate the image
    cv::Mat rotate_image;
    cv::rotate(image, rotate_image, 0);
    cv::imwrite("/users/gaoyingqiang/Desktop/data/task1/rotate_image.jpg", rotate_image);

    // resize 127*105 to 127*127
    cv::Mat padding_image;
    int top, bottom, left, right;
    int borderType;
    borderType = CV_HAL_BORDER_REPLICATE;
    cv::Scalar value;
    cv::RNG rng(12345);

    bottom = (int)(0.21*image.rows);

    value = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
    cv::copyMakeBorder(image, padding_image, 0, bottom, 0, 0, borderType, value);
    cv::imwrite("/users/gaoyingqiang/Desktop/data/task1/padding_image.jpg", padding_image);

    // resize to 640*640
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(640,640), 0, 0, CV_INTER_LINEAR);
    cv::imwrite("/users/gaoyingqiang/Desktop/data/task1/resize.jpg",resize_image);
    //cv::namedWindow("resize_image", CV_WINDOW_NORMAL);
    //cv::imshow("resize_image", resize_image);

    // specify parameters: winSize, blockSize, blockStride, cellSize, nbins
    std::vector<float> descriptors;
    cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
    // compute HOG descriptors
    hog.compute(resize_image, descriptors);
    cout<<"size of HOG:"<<descriptors.size()<<endl;
    // visualization
    visualizeHOG(resize_image, descriptors, hog, 5);

    */


    /*
    // task 2 ///////////////decision tree classifier///////////////

    //prepare the labels for training
    int  train_samples_number[6] = {49,67,42,53,67,110} ; //图片个数
    cv::Mat train_labels;
    for(int i=0;i<6;i++){
        train_labels.push_back(cv::Mat::ones(train_samples_number[i],1,CV_32S)*i);
    }cout<<endl;

    //prepare the data for training
    std::vector<float> train_descriptor;
    cv::Mat train_descriptors;
    cv::Mat train_resize_image;

    for(int i=0;i<6;i++){
        cv::String train_path = "/users/gaoyingqiang/Desktop/data/task2/train/0"+std::to_string(i)+"/";

        for(int j=0;j<train_samples_number[i];j++){
            cv::Mat image = cv::imread(train_path + std::to_string(j) +".jpg", 1);
            // resize to 640*640
            cv::resize(image,train_resize_image,cv::Size(640,640),0,0,CV_INTER_LINEAR);
            // 计算HOG descriptor
            cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
            hog.compute(train_resize_image,train_descriptor);
            cv::Mat descriptor_mat = cv::Mat(train_descriptor).t();
            train_descriptors.push_back(descriptor_mat);
        }
    }
    //cout<<train_descriptors.size()<<endl;

    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_descriptors, cv::ml::ROW_SAMPLE, train_labels);

    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();

    dtree->setMaxDepth(10);
    dtree->setMinSampleCount(10);
    //dtree->setRegressionAccuracy(0.01f);
    //dtree->setUseSurrogates(false);
    dtree->setMaxCategories(6);
    dtree->setCVFolds(0);
    //dtree->setUse1SERule(true);
    //dtree->setTruncatePrunedTree(true);
    //dtree->setPriors(cv::Mat());
    dtree->train(tData);
    cout<<"decision tree training done."<<endl;
    cout<<endl;
    std::string save_dtree{"/users/gaoyingqiang/Desktop/data/task2/train/trained_dtree.xml"};
    dtree->save(save_dtree);


    //predict
    cout<<"predict decision tree starts."<<endl;
    cout<<endl;
    int counter = 0;
    for(int i=0;i<6;i++){
        cv::String test_path = "/users/gaoyingqiang/Desktop/data/task2/test/0"+std::to_string(i)+"/";

        int  test_samples_number = 10 ; //图片个数

        //prepare the data forcout<<endl; testing
        std::vector<float> test_descriptor;
        cv::Mat test_descriptors;
        cv::Mat test_resize_image;

        for(int j=0;j<test_samples_number;j++){
            cv::Mat image = cv::imread(test_path + std::to_string(j) +".jpg", 1);
            // resize to 640*640 && a[i]==a[j]
            cv::resize(image,test_resize_image,cv::Size(640,640),0,0,CV_INTER_LINEAR);
            // 计算HOG descriptor
            cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
            hog.compute(test_resize_image,test_descriptor);
            cv::Mat descriptor_mat = cv::Mat(test_descriptor).t();
            test_descriptors.push_back(descriptor_mat);
        }
        // predict

        cv::Mat test_result;
        std::string load_dtree{"/users/gaoyingqiang/Desktop/data/task2/train/trained_dtree.xml"};
        cv::Ptr<cv::ml::DTrees> Dtree = cv::ml::DTrees::load(load_dtree);

        Dtree->predict(test_descriptors,test_result);


        cv::Mat test_labels;
        test_labels.push_back(cv::Mat::ones(test_samples_number,1,CV_32FC1)*i);

        for(int k=0;k<test_samples_number;++k) {
            float value1 = ((float *) test_labels.data)[k];
            float value2 = ((float *) test_result.data)[k];
            fprintf(stdout, "actual class: %f, expected class: %f\n", value1, value2);
            if (int(value1) == int(value2)) {
                ++counter;
            }
        }

    }
    cout<<endl;
    fprintf(stdout, "total classification accuracy using decision tree: %f\n", counter*1.f/60);
    cout<<endl;

    */




    // task 2 ///////////////random forest classifier///////////////

    cout << endl;
    cout << "random forest starts." << endl;
    cout << endl;

    int num_of_trees = 100;
    randomForest Forest[num_of_trees];
    int correct_class = 15;

    for (int t = 0; t < num_of_trees; t++) {
        srand((unsigned) time(nullptr));
        // size of subset of training data. each class use 1/3 data to train
        int subset_train[6] = {17, 22, 14, 17, 22, 36};
        int range_train[6] = {49, 67, 42, 53, 67, 110};
        int *rand_index;

        std::vector<float> train_descriptor;
        cv::Mat train_descriptors;
        cv::Mat train_resize_image;

        //prepare labels
        cv::Mat train_labels;

        for (int i = 0; i < 6; i++) {
            rand_index = randperm(subset_train[i], 0, range_train[i] - 1); //减1是为了防止抽到正好和range一样大的下标，从而导致访问 不存在的图像

            train_labels.push_back(cv::Mat::ones(subset_train[i], 1, CV_32S) * i);

            cv::String train_path = "/users/gaoyingqiang/Desktop/data/task2/train/0" + std::to_string(i) + "/";
            for (int j = 0; j < subset_train[i]; j++) {
                cv::Mat sub_train;
                sub_train = cv::imread(train_path + std::to_string(rand_index[j]) + ".jpg", 1);

                cv::Mat resize_image;
                cv::resize(sub_train, resize_image, cv::Size(640, 640), 0, 0, CV_INTER_LINEAR);
                // 计算HOG descriptor
                cv::HOGDescriptor hog(cv::Size(640, 640), cv::Size(80, 80), cv::Size(80, 80), cv::Size(80, 80), 9);
                hog.compute(resize_image, train_descriptor);
                cv::Mat descriptor_mat = cv::Mat(train_descriptor).t();
                train_descriptors.push_back(descriptor_mat);
            }
            delete[] rand_index;
        }

        cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_descriptors, cv::ml::ROW_SAMPLE,
                                                                     train_labels);
        cv::Ptr<cv::ml::DTrees> random_forest = Forest[t].creatRF();
        Forest[t].trainRF(random_forest, tData);
        cout << "training for " + std::to_string(t) + ".th tree is done." << endl;

        //prepare the data for testing
        std::vector<float> test_descriptor;
        cv::Mat test_descriptors;
        cv::Mat test_resize_image;
        int test_samples_number = 10;
        cv::Mat test_labels;

        for (int i = 0; i < 6; i++) {
            test_labels.push_back(cv::Mat::ones(test_samples_number, 1, CV_32FC1) * i);
            for (int j = 0; j < test_samples_number; j++) {
                cv::String test_path = "/users/gaoyingqiang/Desktop/data/task2/test/0" + std::to_string(i) + "/";
                cv::Mat image = cv::imread(test_path + std::to_string(j) + ".jpg", 1);
                // resize to 640*640
                cv::resize(image, test_resize_image, cv::Size(640, 640), 0, 0, CV_INTER_LINEAR);
                // 计算HOG descriptor
                cv::HOGDescriptor hog(cv::Size(640, 640), cv::Size(80, 80), cv::Size(80, 80), cv::Size(80, 80), 9);
                hog.compute(test_resize_image, test_descriptor);
                cv::Mat descriptor_mat = cv::Mat(test_descriptor).t();
                test_descriptors.push_back(descriptor_mat);
            }
        }

        // predict
        cv::Mat test_result;
        random_forest->predict(test_descriptors, test_result);
        cout << "predicting for " + std::to_string(t) + ".th tree is done." << endl;

        std::ofstream outFile;
        outFile.open("/users/gaoyingqiang/Desktop/data/task2/rftree_" + std::to_string(t) + "_result.txt");

        for (int i = 0; i < test_samples_number * 6; i++) {
            outFile << ((float *) test_result.data)[i] << endl;
        }
        outFile.close();

        int counter = 0;

        for (int k = 0; k < test_samples_number * 6 - 1; ++k) {
            float value1 = ((float *) test_labels.data)[k];
            float value2 = ((float *) test_result.data)[k];
            fprintf(stdout, "actual class: %f, expected class: %f\n", value1, value2);
            if (int(value1) == int(value2)) {
                ++counter;
            }
        }

        fprintf(stdout, "total classification accuracy: %f\n", counter * 1.f / (test_samples_number * 6));
        cout << endl;
    }


    int test_samples_number = 10;
    float vote_all_trees[test_samples_number * 6][num_of_trees];
    vote_all_trees[test_samples_number * 6][num_of_trees] = {0};
    int tree_index = 0;
    int image_index = 0;

    for (int i = 0; i < num_of_trees; i++) {
        std::ifstream rftree;
        rftree.open("/users/gaoyingqiang/Desktop/data/task2/rftree_" + std::to_string(i) + "_result.txt");
        if (!rftree) {
            cout << "tree not found." << endl;
            exit(1);
        }

        while (!rftree.eof()) {
            rftree >> vote_all_trees[image_index][tree_index];
            if (image_index >= test_samples_number * 6 || tree_index >= num_of_trees) {
                break;
            }
            image_index = image_index + 1;
        }
        tree_index = tree_index + 1;
        image_index = 0;
        rftree.close();
    }

    // vote

    cv::Mat vote_labels;
    for (int i = 0; i < num_of_trees; i++) {
        vote_labels.push_back(cv::Mat::ones(test_samples_number, 1, CV_32FC1) * i);
    }

    // vote_one_tree 100棵树的结果两两比较,统计每一棵树的预测结果在100棵树里出现了多少次
    float vote_one_tree[num_of_trees];

    for (int i = 0; i < test_samples_number * 6; i++) {
        for (int j = 0; j < num_of_trees; j++) {
            vote_one_tree[j] = vote_all_trees[i][j];
        }


        int vote_index = vote_for_result(vote_one_tree, num_of_trees);
        //cout << "the classifaction of " + std::to_string(i) + ".th image is " +std::to_string(vote_one_tree[vote_index]);

        if (((float *) vote_labels.data)[i] == vote_one_tree[vote_index]) {
            correct_class++;
        }

    }
    cout << endl;
    fprintf(stdout, "total classification accuracy using random forest: %f\n", correct_class * 1.f / (test_samples_number * 6));


    return 0;

}






