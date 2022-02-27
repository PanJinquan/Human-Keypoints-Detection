#!/usr/bin/env bash
# 配置docker
#docker run --rm --volume=$(pwd):/opt/TNN/tools/onnx2tnn/onnx-converter/workspace  -it tnn-convert:latest  /bin/bash
#cd /opt/TNN/tools/onnx2tnn/onnx-converter
#bash workspace/utils/convert_tools/onnx2tnn.sh

# 配置模型路径
#model_name="mode-face-person-320-320"
model_name="rfb1.0_face_person_300_300"
onnx_path="workspace/data/pretrained/onnx/"$model_name".onnx"
sim_onnx_path="workspace/data/pretrained/onnx/"$model_name"_sim.onnx"
tnn_model="workspace/data/pretrained/tnn/"


# https://github.com/daquexian/onnx-simplifier
# pip3 install onnx-simplifier
# pip install --upgrade onnx
# python -m onnxsim path/to/src.onnx path/to/src_sim.onnx 0(不做check) --input-shape 1,112,112,3
python3 -m onnxsim  \
    $onnx_path \
    $sim_onnx_path \
    0 \
    --input-shape 1,3,300,300

onnx_path=$sim_onnx_path


# https://github.com/Tencent/TNN/blob/master/doc/cn/user/onnx2tnn.md
#python3 converter.py onnx2tnn \
#    $onnx_path  \
#    -optimize \
#    -v=v3.0 \
#    -o $tnn_model \
#    -align \
#    -input_file in.txt \
#    -ref_file ref.txt

#
#python3 onnx2tnn.py \
#    $onnx_path \
#    -version=v3.0 \
#    -optimize=1 \
#    -half=0 \
#    -o $tnn_model \
#####################################################
#参数说明：
#-version
#模型版本号，便于后续算法进行跟踪
#-optimize
#1（默认，开启）: 用于对模型进行无损融合优化，，如BN+Scale等f融合进Conv层；
#0 ：如果融合报错可以尝试设为此值
#-half
#1: 转为FP16模型存储，减小模型大小。
#0（默认，不开启）: 按照FP32模型存储。
#Note: 实际计算是否用FP16看各个平台特性决定，移动端GPU目前仅支持FP16运算
#-o
#output_dir : 指定 TNN 模型的存放的文件夹路径，该文件夹必须存在
#-input_shape
#模型输入的 shape，用于模型动态 batch 的情况
#####################################################