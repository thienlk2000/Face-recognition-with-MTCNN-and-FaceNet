# Face-recognition-with-MTCNN-and-FaceNet
 
 This repo implements face recognition with MTCNN and FacNet. This repo use python package pytorch-face to use pretrained model of MTCNN and facenet
 
 ## MTCNN
 
 MTCNN is a deeplearning model consists of convolutional layer and can produce cordinates of face bounding boxes and 5 landmark on the face
 
 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/utils/MTCNN.png)
 
 MTCNN have 3 sub network:
 - P-Net can detect multi bouding boxes and multi face in the image because we input this model feature pyramid of image
 - R-Net use bouding box from P-Net crop the patch of images and refine the bouding box
 - O-Net continue refine the bouding box from R-Net and produce 5 landmark of face
 
 ## FaceNet:
 Use model InceptionResnet to produce 512-dims vector represents each face
 We use this 512-dims vector from FaceNet and train neural output layer to classify N_class face based on dataset
 
 ## 1.Dataset
 Dataset directory contain folders corresponding to each class of dataset
 
 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/utils/dataset.JPG)
 
 ## 2.Train model
 First pass iamges through MTCNN to have bouding boxes of face then crop bouding boxes from images and use that sub images to train classification model (FaceNet + output layer) 
 Train model on your dataset by specifying image folder, model file name
```bash
python train.py dataset facenet.pt
```
when training finish we have facenet.pt model weight to detect new images
 ## 3.Detect
 We can detect from image folder or from video and save result by specifying source and dest of data, file weight model, file text contains class names
 ```bash
python detect_image.py data_test data_test_result facenet.pt classes.txt
python detect_video.py video.mp4 result.mp4 facenet.pt classes.txt 
python detect_video.py 0 result.mp4 facenet.pt classes.txt # if use webcam 
```
 ## 4.Result
 We only train model on serveral images per person and Model can detect quite good
- Ben Afflek

 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/test_result/ben_afflek_1.jpg)

- Elton John

 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/test_result/elton_john_1.jpg)
 
- Jerry Seinfeld

 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/test_result/jerry_seinfeld_1.jpg)
 
- Madonna

 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/test_result/madonna_1.jpg)
 
- Mindy Kaling

 ![file](https://github.com/thienlk2000/Face-recognition-with-MTCNN-and-FaceNet/blob/main/test_result/mindy_kaling_1.jpg)
 
 
 
