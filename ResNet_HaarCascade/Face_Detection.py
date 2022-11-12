import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

print(torch.cuda.is_available())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Load Resnet152 Model
PATH = './ResNet152Mod_mask_model2.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network = models.resnet152()
img_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Adjust FC layer
network.fc = torch.nn.Linear(network.fc.in_features,3)
network.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
network.to(device)
network.eval()

#Open webcam via OpenCV
font = cv2.FONT_HERSHEY_COMPLEX
webcam = cv2.VideoCapture(0)
fps = int(webcam.get(cv2.CAP_PROP_FPS))
key = ord('0')

while key != ord('q'):
    check,frame = webcam.read()
    height,width = frame.shape[:2]
    key = cv2.waitKey(fps)
    img = Image.fromarray(frame)
    img = np.asarray(img)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        imgP = img[y:y + h, x:x + w]
        imgP = Image.fromarray(np.uint8(imgP))
        imgP = img_transforms(imgP)
        imgP = imgP.unsqueeze(0)
        indata = imgP.to(device)
        output = network(indata)
        _, pred = torch.max(output, 1)

        if (pred[0]==0):
            label = "Incorrect mask!"
            bgr = (0, 255, 255)
        elif (pred[0]==1):
            label = "Wearing mask!"
            bgr = (0, 255, 0)
        elif (pred[0]==2):
            label = "No mask!"
            bgr = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        cv2.putText(frame, str(label), (x, y), 1, 1.3, bgr, 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
webcam.release()
print("Camera off.")
cv2.destroyAllWindows()