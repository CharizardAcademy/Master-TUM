int num_of_trees = 10;
randomForest Forest[num_of_trees];

for (int t = 0; t < num_of_trees; t++) {
srand((unsigned) time(NULL));
// size of subset of training data. each class use 1/3 data to train
int subset_train[6] = {17, 22, 14, 17, 22, 36};
int range_train[6] = {49, 67, 42, 53, 66, 110};
int *rand_index;

std::vector<float> train_descriptor;
cv::Mat train_descriptors;
cv::Mat train_resize_image;

//prepare labels
cv::Mat train_labels;

for (int i = 0; i < 6; i++) {
rand_index = randperm(subset_train[i], 0, range_train[i]-1); //减1是为了防止抽到正好和range一样大的下标，从而导致访问 不存在的图像

train_labels.push_back(cv::Mat::ones(subset_train[i],1,CV_32S)*i);

cv::String train_path = "/home/highstars1/桌面/data/task2/train/0" + std::to_string(i) + "/";
for (int j = 0; j < subset_train[i]; j++) {
cv::Mat sub_train;
sub_train = cv::imread(train_path + std::to_string(rand_index[j]) + ".jpg", 1);
//cv::namedWindow("resize_image", CV_WINDOW_NORMAL);
//cv::imshow("resize_image", sub_train);
//cv::waitKey(1000);

cv::Mat resize_image;
cv::resize(sub_train, resize_image, cv::Size(640, 640), 0, 0, CV_INTER_LINEAR);
// 计算HOG descriptor
cv::HOGDescriptor hog(cv::Size(640, 640), cv::Size(80, 80), cv::Size(80, 80), cv::Size(80, 80), 9);
hog.compute(resize_image, train_descriptor);
cv::Mat descriptor_mat(cv::Mat(train_descriptor).t());
train_descriptors.push_back(descriptor_mat);
}
delete[] rand_index;
}

cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_descriptors, cv::ml::ROW_SAMPLE,train_labels);
Forest[t].creatRF();
Forest[t].trainRF(Forest[t].rftree, tData);
//std::string save_rftree{"/home/highstars1/桌面/data/task2/train/trained_rftree_"+std::to_string(t)+".xml"};
//Forest[t].rftree->save(save_rftree);
cout << "training for " + std::to_string(t) + ".th tree is done." << endl;

//prepare the data for testing
std::vector<float> test_descriptor;
cv::Mat test_descriptors;
cv::Mat test_resize_image;
int test_samples_number = 10;
cv::Mat test_labels;

for(int i=0;i<6;i++){
test_labels.push_back(cv::Mat::ones(test_samples_number,1,CV_32FC1)*i);
for(int j=0;j<test_samples_number;j++){
cv::String test_path = "/home/highstars1/桌面/data/task2/test/0" + std::to_string(i) + "/";
cv::Mat image = cv::imread(test_path + std::to_string(j) +".jpg", 1);
// resize to 640*640
cv::resize(image,test_resize_image,cv::Size(640,640),0,0,CV_INTER_LINEAR);
// 计算HOG descriptor
cv::HOGDescriptor hog(cv::Size(640,640), cv::Size(80,80), cv::Size(80,80), cv::Size(80,80), 9);
hog.compute(test_resize_image,test_descriptor);
cv::Mat descriptor_mat(cv::Mat(test_descriptor).t());
test_descriptors.push_back(descriptor_mat);
}
}

// predict
cv::Mat test_result;
Forest[t].rftree->predict(test_descriptors,test_result);
cout<<"predicting for " + std::to_string(t) + ".th tree is done."<<endl;

std::ofstream outFile;
outFile.open("/home/highstars1/桌面/data/task2/rftree_"+std::to_string(t)+"_result.txt");

for(int i=0;i<test_samples_number*6;i++){
outFile<<((float*)test_result.data)[i]<<endl;
}
outFile.close();

int counter = 0;

for(int k=0;k<test_samples_number*6-1;++k){
float value1 = ((float*)test_labels.data)[k];
float value2 = ((float*)test_result.data)[k];
fprintf(stdout, "actual class: %f, expected class: %f\n", value1, value2);
if(int(value1)==int(value2)){
++counter;
}
}
fprintf(stdout, "total classification accuracy: %f\n", counter*1.f/(test_samples_number*6));

}

//cout<<"training done."<<endl;

// vote for the final classifaction
int test_samples_number = 10;
float vote_all_trees[test_samples_number*6][num_of_trees];
int tree_index = 0;
int image_index = 0;
for(int i=0;i<num_of_trees;i++){
std::ifstream rftree;
rftree.open("/home/highstars1/桌面/data/task2/rftree_"+std::to_string(i)+"_result.txt");
if(!rftree){
cout<<"tree not found."<<endl;
}
while(!rftree.eof()){
rftree>>vote_all_trees[image_index][tree_index];
if(image_index>=test_samples_number*6||tree_index>=num_of_trees){
break;
}
image_index = image_index + 1;
}
tree_index = tree_index + 1;
rftree.close();

}

for(int i=0;i<60;i++){
cout<<vote_all_trees[i][1]<<"\t";
}