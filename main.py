import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from deep_sort.deep_sort import DeepSort

device = select_device('')
model = DetectMultiBackend('weights/yolov5s.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1, 3, 640, 640))

tracker = DeepSort(model_path='deep_sort/deep/checkpoint/ckpt.t7')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, 0.4, 0.5)

    dets = []
    if pred[0] is not None and len(pred[0]):
        for *xyxy, conf, cls in pred[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2 - x1, y2 - y1, conf.item()])

    tracks = tracker.update_tracks(dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Object Detection & Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
