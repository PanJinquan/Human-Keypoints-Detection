#ifndef ACTION_STATUS_H
#define ACTION_STATUS_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <stdio.h>
#include <queue>
#define  millisecond 1000000
#define DEBUG_TIME(time_) auto time_ =std::chrono::high_resolution_clock::now()
#define RUN_TIME(time_)  (double)(time_).count()/millisecond
#define DEBUG_PRINT(...)  printf( __VA_ARGS__); printf("\n")

using namespace std;

template<typename _Tp>
vector<_Tp> convertMat2Vector(const cv::Mat &mat)
{
    return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}


class Monitor
{
public:
    Monitor(char *request_id, int window_size = 4, float dist_thresh = 0.05f, float up_thresh = 0.2f, int dst_w = 96, int dst_h = 96, int top_k = 400);
    ~Monitor();
    int getStatus(cv::Mat &frame, int frame_id, bool is_save = false, char *output_dir = (char*) "./output");
    bool is_upload=false;

private:
    int processInput(cv::Mat &src, cv::Mat &dst);
    float getSimilarity(cv::Mat &src_1, cv::Mat &src_2);
    bool isUpload(int label);

private:
    int frame_id;
    std::string req_id;
    cv::Mat cur_frame;
    cv::Mat last_frame;
    cv::Mat upload_frame;
    int window_sz;
    float distThresh;
    float uploadThresh;
    int target_w;
    int target_h;
    std::vector<int> label_buffer;
    int flag[4] = {1, 0, 0, 0};
    int flag_sz = 4;
    int topK;
    int num_frame = 0;
};
#endif //ACTION_STATUS_H