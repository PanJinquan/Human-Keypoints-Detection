#include "ActionStatus.h"



Monitor::Monitor(char *request_id, int window_size, float dist_thresh, float up_thresh, int dst_w, int dst_h, int top_k)
{
    req_id = request_id;
    window_sz = window_size;
    distThresh = dist_thresh;
    uploadThresh = up_thresh;
    target_w = dst_w;
    target_h = dst_h;
    topK = top_k;
}

Monitor::~Monitor()
{
    label_buffer.clear();
    // upload_frame.release();
    // cur_frame.release();
    // last_frame.release();
};

int Monitor::processInput(cv::Mat &src, cv::Mat &dst)
{
	DEBUG_TIME(T0);
	cv::Mat rgb, rs_rgb;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2GRAY);
    cv::resize(rgb, rs_rgb, cv::Size(target_w, target_h), cv::INTER_NEAREST);
    cv::GaussianBlur(rs_rgb, dst, cv::Size(3, 3), 0);
    dst.convertTo(dst,CV_32F,1/255.0,0);
	DEBUG_TIME(T1);
	DEBUG_PRINT("--> processInput:%3.3f", RUN_TIME(T1 - T0));
    return 0;
};

float Monitor::getSimilarity(cv::Mat &src_1, cv::Mat &src_2)
{
    int tok=400;
    // get diff map
	DEBUG_TIME(T0);
    cv::Mat diff_map=cv::abs(src_2-src_1);
	DEBUG_TIME(T1);
    cv::imshow("diff_map",diff_map);
    DEBUG_PRINT("--> diff_map:%3.3f", RUN_TIME(T1 - T0));
	DEBUG_TIME(T2);
    vector<float> diff_buffer = convertMat2Vector<float>(diff_map);
    //cv::sort(diff_map,diff_map,cv::CV_SORT_EVERY_ROW + cv::CV_SORT_ASCENDING);
    //sort(diff_buffer.begin(), diff_buffer.end());//
	sort(diff_buffer.rbegin(), diff_buffer.rend());//
    DEBUG_TIME(T3);
	DEBUG_PRINT("--> sort:%3.3f", RUN_TIME(T3 - T2));
    float dist = 0.0f;
    for (int i = 0; i < tok; i++)
        dist += diff_buffer[i];
    dist /= (tok);
	DEBUG_TIME(T4);
	DEBUG_PRINT("--> tok:%3.3f", RUN_TIME(T4 - T3));
    return dist;
};

bool Monitor::isUpload(int label)
{
    bool is_upload = false;

    if (label_buffer.size() == window_sz)
        label_buffer.erase(label_buffer.begin());
    label_buffer.push_back(label);
    int cnt = 0;
    if (label == 0)
    {
        if (label_buffer.size() < flag_sz)
        {
            return is_upload;
        }

        for (int i = 0; i < label_buffer.size() - flag_sz + 1; i++)
        {
            if (label_buffer[i] == 1 && label_buffer[i + 1] == 0 && label_buffer[i + 2] == 0 && label_buffer[i + 3] == 0)
            {
                cnt++;
            }
        }
        is_upload = (cnt == 1);
        if (is_upload)
        {
            float dist = getSimilarity(upload_frame, cur_frame);
            is_upload = dist >= uploadThresh;
        }
    }
    return is_upload;
}

int Monitor::getStatus(cv::Mat &frame, int frame_id, bool is_save, char *output_dir)
{
    float dist = 0;
    int label = 0;
    int ret = processInput(frame, cur_frame);
	num_frame = frame_id;
    if (num_frame == 1)
    {
        last_frame = cur_frame;
        upload_frame = cur_frame.clone();
        dist = 0;
    }
    else
    {
        dist = getSimilarity(last_frame, cur_frame);
    }

    label = dist < distThresh ? 0 : 1;
	DEBUG_PRINT("dist: %3.3f",dist );
    is_upload = isUpload(label);
    if (is_upload)
    {
		upload_frame = cur_frame.clone();
		if (is_save)
        {
			char save_path[256];
            sprintf(save_path, "%s%04d_upload.jpg", output_dir, frame_id);
            DEBUG_PRINT("upload frame: %d\n",save_path);
            cv::imwrite(save_path, frame);
        }
    }

    last_frame = cur_frame;
    return label;
}