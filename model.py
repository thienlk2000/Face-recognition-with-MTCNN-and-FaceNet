import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
class FaceNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.embedding  = InceptionResnetV1(
                pretrained='vggface2')

        for parameter in self.embedding.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(512, n_classes)
    def forward(self, x):
        self.embedding.eval()
        x = self.embedding(x)
        x = self.classifier(x)
        return x
    