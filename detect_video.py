from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torchvision import transforms as T
import numpy as np
import os
import glob
from utils.train_model import train
from model import FaceNet
from PIL import Image
import cv2
import argparse

def plot_box(img, box, name):
    cv2.rectangle(img, box[0:2].astype(np.int16), box[2:4].astype(np.int16), color=(0,0,255), thickness=2)
    cv2.putText(img, name, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255))
    return img
def tune_box(box, size, to_size):
    box[:, 0] = box[: ,0] * to_size[1] / size[0]
    box[:, 1] = box[: ,1] * to_size[0] / size[1]
    box[:, 2] = box[: ,2] * to_size[1] / size[0]
    box[:, 3] = box[: ,3] * to_size[0] / size[1]
    return box

parser = argparse.ArgumentParser()
parser.add_argument("video_file", help='source video file to detect face')
parser.add_argument("save_file", help='save video')

parser.add_argument("model_name", help='model file name')
parser.add_argument("class_file", help="class name to detect face")
args = parser.parse_args()

video_file = args.video_file
data_save = args.save_file
model_name = args.model_name
class_file = args.class_file


with open(class_file, 'r') as f:
     class_name = f.read().splitlines()

num_classes = len(class_name) 


device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=True
)
model = FaceNet(num_classes)
model.load_state_dict(torch.load(model_name))

transform_1 = T.Resize((512, 512))
if video_file == "0":
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(video_file)

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(data_save, fourcc, 20.0, (int(width),  int(height)))


while video.isOpened():
    ret, frame = video.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    original_size = frame.shape
    reshape_size = (512, 512)
    img = cv2.resize(frame, (512, 512))
    batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
    if batch_boxes is None:
        continue
    faces = mtcnn.extract(img, batch_boxes, save_path=None)
    boxes = tune_box(batch_boxes, reshape_size, original_size)
    for box, face in zip(boxes, faces):
        face = face.unsqueeze(0)
        score = model(face)
        y_pred = score.argmax(dim=1)[0].item()
        name = class_name[y_pred]
        plot_box(frame, box, name)
    cv2.imshow("result", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()