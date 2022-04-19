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
parser.add_argument("root_dir", help='directory contains data to detect face')
parser.add_argument("save_dir", help='save dir to save result')

parser.add_argument("model_name", help='model file name')
parser.add_argument("class_file", help="class name to detect face")
args = parser.parse_args()

data_dir = args.root_dir
data_save = args.save_dir
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
# transform_2 = 

# data_dir = 'data_test'
# data_save = 'data_test_result'
img_files = glob.glob(data_dir + '/*.jpg')
images = [(Image.open(img_file)) for img_file in img_files]
images_transformed = [transform_1(Image.open(img_file)) for img_file in img_files]
batch_boxes, batch_probs, batch_points = mtcnn.detect(images_transformed, landmarks=True)
faces = mtcnn.extract(images_transformed, batch_boxes, save_path=None)
for i,(img_file, image_transformed, boxes, faces) in enumerate(zip(img_files, images_transformed, batch_boxes, faces)):
    img_name = os.path.basename(img_file)
    img = cv2.imread(img_file)
    original_size = img.shape
    reshape_size = (512, 512)
    if boxes is None:
        cv2.imwrite(data_save + f'/result{i}.jpg', img)
        continue

    boxes = tune_box(boxes, reshape_size, original_size)
    for box, face in (zip(boxes, faces)):
        print(box)
        face = face.unsqueeze(0)
        score = model(face)
        y_pred = score.argmax(dim=1)[0].item()
        name = class_name[y_pred]
        plot_box(img, box, name)
    cv2.imwrite(data_save + '/' +img_name, img)





