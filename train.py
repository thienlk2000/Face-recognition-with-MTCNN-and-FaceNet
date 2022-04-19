from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os
from utils.train_model import train
from model import FaceNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root_dir", help='directory contains data with each class in correspond directory')
parser.add_argument("model_name", help='file name to save model')
args = parser.parse_args()



model_name = args.model_name
data_dir = args.root_dir
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    batch_size=5,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# resnet = InceptionResnetV1(
#     classify=True,
#     pretrained='vggface2',
#     num_classes=len(dataset.class_to_idx)
# ).to(device)

# test = torch.randn(2,3,160,160)
# print(resnet(test).shape)

facenet = FaceNet(len(dataset.class_to_idx))

optimizer = optim.Adam(facenet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()

train(facenet, loss_fn, optimizer, scheduler, train_loader, val_loader, model_name, device, epoch=5)