#!/usr/bin/env bash
if [ ! -d "build/" ];then
  mkdir "build"
else
  echo "exist build"
fi
cd build
#cmake ..
#cmake .. -DCMAKE_BUILD_TYPE=Release -NCNN_AVX2=ON -NCNN_OPENMP=OFF
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

sleep 1
./Demo

# ./Demo /home/dm/panjinquan3/git_project/StatusMonitor/data/finger_video.mp4 1 1 ./output

