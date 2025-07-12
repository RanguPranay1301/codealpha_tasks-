# Object Detection and Tracking

This project performs real-time object detection and tracking using YOLOv5 and Deep SORT.

## Setup Instructions

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Clone YOLOv5:
```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

3. Clone Deep SORT:
```
git clone https://github.com/nwojke/deep_sort.git
```

4. Download YOLOv5 weights to `weights/yolov5s.pt`:
```
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P weights/
```

5. Run the project:
```
python main.py
```
