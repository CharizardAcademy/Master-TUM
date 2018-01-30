#ifndef RF_H
#define RF_H

//#include <vector>
#include <memory>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
//using namespace std;

void Swap(int &a, int &b);

void PickMRandomly(int a[], int n, int m);

void printArray(int arr[], int size);

class RandomForest1
 {
  public:
 //create
     RandomForest1(int parraysize);
 //train
     void train(Mat data, Mat labels);
 //predict
     Mat predict(Mat samples);

 private:
    int TreeNumber;
    std::unique_ptr<Ptr<ml::DTrees>[] > model ;
 };

int get_correct_number( std::vector<cv::Rect>& resRects, cv::Rect gt);
int check_truth( std::vector<cv::Rect>& resRects, cv::Rect gt);


inline void nms(
        const std::vector<cv::Rect>& srcRects,
        std::vector<cv::Rect>& resRects,
        float thresh,
        int neighbors = 0
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
        {
            resRects.push_back(rect1);
        }
    }
}



#endif // RF_H
