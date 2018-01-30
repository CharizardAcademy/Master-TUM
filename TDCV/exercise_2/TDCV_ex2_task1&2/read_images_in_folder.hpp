std::vector<cv::Mat> read_images_in_folder(cv::String pattern){
    std::vector<cv::String> fn;
    glob(pattern, fn, false);

    std::vector<cv::Mat> images;
    size_t count = fn.size(); // number of jpg files in images folder
    for (size_t i=0;i<count;i++){
        images.push_back(cv::imread(fn[i]));
        cv::imshow("img", cv::imread(fn[i]));
        cv::waitKey(1000);
    }
    return images;
}
