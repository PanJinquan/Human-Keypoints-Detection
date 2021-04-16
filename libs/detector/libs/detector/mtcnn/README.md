# face landmark-detection
基于MTCNN-ONet的人脸5个关键点检测模型
## 1. 目录结构

```
├── net 
│   ├── onet_landmark.py         # ONet网络模型
│   └── box_utils.py             # ONet依赖的处理函数
├── demo.py           # 人脸关键点检测Demo文件
├── md5sum.txt        # ONet权重MD5文件
├── test.jpg          # 测试图片
├── XMC2-landmark-detection.pth.tar  # ONet权重文件
└── README.md
```
## 2. Platform
- hardware: Intel Core i7-8700 CPU @ 3.20GHz × 12, GPU GeForce RTX 2070 8G
- Python3.6
- Pillow-6.0
- Pytorch-1.0.1
- torchvision-0.2.2
- numpy-1.16.3
- opencv-python 3.4.1



## 3. I/O

```
Input:  输入已经裁剪的RGB人脸图像,size:任意
Output: 人脸landmarks的5个关键点(x,y):type:list,shape=(-1,5,2)
```

## 4. Run a demo

```bash
python demo.py 
```

输出结果

```  
landmarks:[[[37.079113, 62.736397], [80.8125, 55.17576], [67.646545, 84.524025], [48.907936, 115.27969], [83.53348, 108.24533]]]

```

