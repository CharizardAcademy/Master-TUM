//
// Created by highstars1 on 01.01.18.
//
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <opencv2/ml/ml.hpp>


int dtree_train(){
    /*

    for (int j=0;j<6;j++){
        cv::String train_path = "/home/highstars1/桌面/data/task2/train/0"+std::to_string(j)+"/";

        std::vector<float> train_descriptor;
        cv::Mat train_descriptors;
        cv::Mat resize_image;

        for(int i=0;i<train_samples_number[j];i++){
            cv::Mat image = cv::imread(train_path + std::to_string(i) +".jpg", 1);
            // resize to 640*640
            cv::resize(image,resize_image,cv::Size(640,640),0,0,CV_INTER_LINEAR);
            // 计算HOG descriptor
            cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
            hog.compute(resize_image,train_descriptor);
            cv::Mat descriptor_mat(cv::Mat(train_descriptor).t());
            train_descriptors.push_back(descriptor_mat);
        }

        cv::Mat train_labels;
        if(j==0){
            train_labels.push_back(cv::Mat::zeros(train_samples_number[j],1,CV_32FC1));
        }
        else{
            train_labels.push_back(cv::Mat::ones(train_samples_number[j],1,CV_32FC1)*j);
        }

        cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_descriptors, cv::ml::ROW_SAMPLE, train_labels);

        if(j==0){
            cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
            cv::ml::DTrees *Dtree = dtree;

            dtree->setMaxDepth(10);
            dtree->setMinSampleCount(10);
            //dtree->setRegressionAccuracy(0.01f);
            //dtree->setUseSurrogates(false);
            dtree->setMaxCategories(2);
            dtree->setCVFolds(0);
            //dtree->setUse1SERule(true);
            //dtree->setTruncatePrunedTree(true);
            //dtree->setPriors(cv::Mat());
            dtree->train(tData);
            cout<<"training for class " + std::to_string(j)<<endl;
            std::string save_dtree{"/home/highstars1/桌面/data/task2/train/trained_dtree.xml"};
            dtree->save(save_dtree);
        }
        else{
            std::string load_dtree{"/home/highstars1/桌面/data/task2/train/trained_dtree.xml"};
            cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::load(load_dtree);
            dtree->train(tData);
            cout<<"training for class " + std::to_string(j)<<endl;
            std::string save_dtree{"/home/highstars1/桌面/data/task2/train/trained_dtree.xml"};
            dtree->save(save_dtree);
        }

    }

    cout<<"training done."<<endl;



    // predict
    for(int k=0;k<6;k++){
        cv::String test_path = "/home/highstars1/桌面/data/task2/test/0"+std::to_string(k)+"/";
        int  test_samples_number = 10 ; //图片个数

        std::vector<float> test_descriptor;
        cv::Mat test_descriptors;
        cv::Mat resize_image;

        for(int i=0;i<test_samples_number;i++){
            cv::Mat image = cv::imread(test_path + std::to_string(i) +".jpg", 1);
            // resize to 640*640
            cv::resize(image,resize_image,cv::Size(640,640),0,0,CV_INTER_LINEAR);
            // 计算HOG descriptor
            cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
            hog.compute(resize_image,test_descriptor);
            cv::Mat descriptor_mat(cv::Mat(test_descriptor).t());
            test_descriptors.push_back(descriptor_mat);
        }

        cv::Mat test_result;
       //cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(test_descriptors, cv::ml::ROW_SAMPLE);

        std::string load_dtree{"/home/highstars1/桌面/data/task2/train/trained_dtree.xml"};
        cv::Ptr<cv::ml::DTrees> Dtree = cv::ml::DTrees::load(load_dtree);

        Dtree->predict(test_descriptors,test_result);

        cv::Mat test_labels;
        if(k==0){
            test_labels.push_back(cv::Mat::zeros(test_samples_number,1,CV_32FC1));
        }
        else{
            test_labels.push_back(cv::Mat::ones(test_samples_number,1,CV_32FC1)*k);
        }

        int counter = 0;
        for(int i=0;i<test_samples_number;++i){
            float value1 = ((float*)test_labels.data)[i];
            float value2 = ((float*)test_result.data)[i];
            fprintf(stdout, "actual class: %f, expected class: %f\n", value1, value2);
            if(int(value1)==int(value2)){
                ++counter;
            }
        }
        fprintf(stdout, "classification accuracy: %f\n", counter*1.f/test_samples_number);
    }
*/
}
