#include "rf.h"

void Swap(int &a, int &b){
    int t = a;
    a = b;
    b = t;
}

void PickMRandomly(int a[],int n, int m){
    srand((unsigned)time(NULL));
    for(int i=0;i<m;++i)
    {
        int j = rand()%(n-i)+i;
        Swap(a[i],a[j]);
    }
}

void printArray(int arr[], int size) {
    for ( int i = 0; i < size; i++ ) {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;
}

RandomForest1::RandomForest1(int parraysize)
    : TreeNumber{parraysize}, model{new Ptr<ml::DTrees> [TreeNumber]}
{
    for(int i=0; i < TreeNumber; i++){
        model[i] = ml::DTrees::create();
        model[i]->setMaxDepth(20);
        model[i]->setCVFolds(0);
    }
}

int get_correct_number( std::vector<cv::Rect>& resRects, cv::Rect gt)
{
    int number = 0;
    if(resRects.size()==0)
        return 0;
    else{
        for(int i=0; i < resRects.size(); ++i){
            float intArea = (resRects[i] & gt).area();
            float unionArea = resRects[i].area() + gt.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap >= 0.3)
                number++;
        }
        return number;
    }
}

int check_truth( std::vector<cv::Rect>& resRects, cv::Rect gt)
{
    int number = 0;
    if(resRects.size()==0)
        return 0;
    else{
        for(int i=0; i < resRects.size(); ++i){
            float intArea = (resRects[i] & gt).area();
            float unionArea = resRects[i].area() + gt.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap >= 0.3)
                number = 1;
        }
        return number;
    }
}


void RandomForest1::train(Mat data, Mat labels)
 {
     //srand((unsigned)time(NULL));
     for(int i=0; i < TreeNumber; i++){
     // train every tree
         //rate 0.7
         int n=labels.size().height, m=int(0.7*labels.size().height);
         int a[n];
         for(int i=0;i<n;++i)
         {
             a[i]=i;
         }
         PickMRandomly(a,n,m);
         //cout << "random variables" <<  endl;
         //printArray(a, m);
         //cout << "n is " << n << " m is " << m << endl;

         Mat data_selected, label_selected;
         for(int i=0;i<m;++i)
         {
             label_selected.push_back(labels.row(a[i]));
             data_selected.push_back(data.row(a[i]));
             //labels.row(a[i]).copyTo(label_selected.row(i+1));
         }
         model[i]->train(ml::TrainData::create(data_selected,ml::ROW_SAMPLE,label_selected));
     }
 }

Mat RandomForest1::predict(Mat samples){

    Mat result;
    model[0]->predict(samples,result);
    int catergory = 4;

    for(int i=1;i<TreeNumber;++i){
       Mat temp_result;
       model[i]->predict(samples,temp_result);
       hconcat(result,temp_result,result);
    }
    result.convertTo(result,CV_32S);

    Mat single_result;
    for(int i=0;i<result.size().height;++i){
        int histo[4] = {0};
        for(int j=0;j<result.size().width;++j){

            int a = result.at<int>(i,j);
            //std::cout << "doning work" << a << std::endl;
            histo[a]++;
        }

        int c=0, max_count=0;
        for(int k=0;k<catergory;k++)
        {
            if(histo[k]>max_count)
            {
                c = k;
                max_count = histo[k];
            }
        }
        float rate = float(max_count)/result.size().width;
        //single_result.push_back(float(c));
        //single_result.push_back(rate);
        Vec2f pair = Vec2f(float(c),rate);
        single_result.push_back(pair);
    }

    return single_result;
}
