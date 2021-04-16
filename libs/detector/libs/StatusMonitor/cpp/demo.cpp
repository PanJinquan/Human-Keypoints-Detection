#include "ActionStatus.h"
#include <stdio.h>
#include <iostream>
using namespace std;
int main(int argc, char **argv)
{

	int detect_freq = 7;
	bool is_show = true;
	bool is_save = true;
	char *save_dir = (char*)"../../data/output/";
	string video_dir = "../../data/finger_video4.mp4";

	Monitor monitor = Monitor(save_dir);
	cv::Mat frame;
	cv::VideoCapture cap;
	cap.open(video_dir);
	if (!cap.isOpened())
		std::cout << "Open video failed.\n";
	int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int numFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
	int fps = int(cap.get(cv::CAP_PROP_FPS));
	int freq = int(fps / detect_freq);

	int label;
	int frame_id = 0;
	bool  is_upload = false;
	while (cap.read(frame))
	{
		frame_id++;
		if (frame_id % freq == 0 || frame_id == 1) {
			label = monitor.getStatus(frame, frame_id, is_save, save_dir);
			is_upload = monitor.is_upload;
			DEBUG_PRINT("frame_id:%d, label:%d, is_upload:%d", frame_id, label, is_upload);
		}
		else {
			is_upload = false;
		}
		cv::imshow("image", frame);
		if (is_upload) {
			cv::waitKey(0);
		}
		cv::waitKey(30);
		std::cout << "\n";
	}
	return 0;
}
