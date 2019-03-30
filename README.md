# Find Face

This project use tensorflow yolo to detect face and SIFT/(dlib face feature) to match face.

## Demo (Find Trump!!)
![demo](https://github.com/FingerRec/find_face/raw/master/output/output_1.png)

## Requirements
> tensorflow-1.8.0
> opencv-3.4.0
> face-recognition-1.2.3
> python3

all these requirements can be installed use below sentence.
### install requirements

```bash
pip install -r requirements.txt
```
## RUN
### 1.mkdir and download pretrained model.

```bash
mkdir pretrained_model
```

download from [dropbox](https://www.dropbox.com/s/bshk79jod7h0y41/yolo_v3_face_detect.pb?dl=0) and put it into dir pretrained_model

### 2.run demo
```bash
python dl_main.py
```
directly or
```bash
python dl_main.py --input1 [path_of_img_1] --input2 [path_of_img_2]
```

* Tip: this code is for my cv project and still be developing.
* Tip: I have crawling one thousand trump images from google and naver. Download them from [dropbox](https://www.dropbox.com/sh/t5u1ra4sef24kq4/AACNc1uHNFJRXlaiUMDFfBxta?dl=0) if need.

## Acknolgment
The code is highly based on and pretrained model is from [YOLOv3-Based](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking).