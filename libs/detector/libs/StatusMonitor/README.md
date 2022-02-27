# StatusMonitor

检测视频流状态,计算连续相邻两帧的相似性,判断当前状态{"运动":1,"静止":0},判断是否需要上传图片

## 1.基本思路
#### 规则1.判断当前帧是`静止`还是`运动`状态

- 1. 一秒抽N帧进行如下处理(detect_freq=7)
- 2. 将图片resize(96, 96),建议采用最近邻插值方式,减小处理时间
- 3. 图像预处理: 将图像转换为灰度图,并进行高斯滤波,去除噪声干扰
- 4. 比较`上一帧图片`与`下一帧(当前帧)`的相似度(距离): 
```
   4.1 计算两帧图片的差异图(两张图的绝对差): diff_map = np.abs(frame1 - frame2) / 255.0
   4.2 将差异图作为特征,按特征值大小进行降序排序,提取tok的差异较大的特征,计算其均方差l2(或者平均绝对误差l1)作为相似距离
```
- 5. 若相似度(距离)小于threshold,则认为图像处理`静止状态0`,或者处于`运动状态1`
- 6. 接口说明:

```python
    def get_frame_similarity(self, frame1, frame2, measurement="l1", isshow=False):
        """
        比较两帧图像的相似度(距离)
        :param frame1: 输入frame1图像
        :param frame2: 输入frame2图像
        :param measurement: 相似度(距离)度量方法,l1: 平均绝对误差作为相似度(距离),l2: 均方差作为相似距离
        :param isshow: <bool> 默认为False,是否显示差异图
        :return: <float>相似度(距离)
        """
````


#### 规则2.判断是否需要上传图片

- 1. 当且仅当前帧图片处于静止状态(label="0")
- 2. 且先出现1次label=1,然后连续3帧的label都是0,则认为需要上传图片(flag="1000")
- 3. 并且当前帧图片与上一次上传的图片不相似时,才需要上传图片
- 4. 接口说明:

```python
    def check_upload_status(self, frame, upload_frame, label, up_threshold=0.001, up_flag="1000"):
        """
        判断是否需要上传图片
        思路: 当且仅当前帧图片处于静止状态(label="0"),且历史label中出现上传图片的信号符(flag="1000"),
              并且当前帧图片与上一次上传的图片不相似时(>up_threshold),才需要上传图片
        :param frame: 当前帧图
        :param upload_frame: 上一次上传的图像
        :param label: 当前label: "0"表示静止,"1"表示运动,
        :param up_threshold: 上传图片的相似阈值,当frame与上一次的upload_frame的相似性距离大于该阈值时,则上传图片
                             否则,认为当前帧与上一次上传的图片非常相似,不需要上传
        :param up_flag: 上传图片的信号符: 默认"1000": 表示先出现1次label=1,然后连续3帧的label都是0,则认为需要上传图片,
                     flag="110000": 表示先出现2次label=1,然后连续4帧的label都是0,则认为需要上传图片
        :return: <bool>: True : 表示需要上传图片
                         False: 表示不需要上传图片
        """

```

## 2.demo

- detect_freq  : 检测频率,1S内检测的帧数,默认1秒检测7帧
- threshold    : 相似度(距离)阈值,用于判断当前状态是: "运动"还是"静止"
- up_threshold : 上传图片的相似阈值,当frame与上一次的upload_frame的相似性距离大于该阈值时,则上传图片
                否则,认为当前帧与上一次上传的图片非常相似,不需要上传
- up_flag      : 上传图片的信号符: 默认"1000": 表示先出现1次label=1,然后连续3帧的label都是0,则认为需要上传图片
- win_size     : 状态检测器的窗口大小,用于记录历史label信息,推荐等于上传图片的信号符up_flag长度
- measurement  : 相似度(距离)度量方法,l1: 平均绝对误差作为相似度(距离); l2: 均方差作为相似度(距离),l1比l2稍微快一点
- 以下是默认的参数配置,推荐使用`config_l1`

```python
# 采用l1平均绝对误差作为相似度(距离)的参数设置
config_l1 = {"detect_freq": 7,
             "threshold": 0.05,
             "up_threshold": 0.2,
             "win_size": 4,
             "up_flag": "1000",
             "measurement": "l1",
             }

# 采用l2均方差作为相似度(距离)的参数设置
config_l2 = {"detect_freq": 7,
             "threshold": 0.0025,
             "up_threshold": 0.2,
             "win_size": 4,
             "up_flag": "1000",
             "measurement": "l2",
}
```


```bash
python demo.py
```

- `data/finger_video`目录下会保存需要上传的图片
