::conda create -n pytorch-py36 python==3.6.7
::conda activate  pytorch-py36
::pip install -r requirements.txt
python run.py --image_file data/test_images
python run.py --video_file data/videos/kunkun_cut.mp4
python run.py --video_file 0