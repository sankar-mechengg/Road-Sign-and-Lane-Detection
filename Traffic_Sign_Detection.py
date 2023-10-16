# This program is to detect traffic or road signs in a real time video stream using YoloV8 model

# Importing the required libraries
import cv2
import numpy as np
import time
import os
import argparse
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from IPython.display import Image as IPyImage

# Ultralytics Check for the latest version
ultralytics.checks()

# Downloading the Dataset from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Rt8xNBTVlFtLkVFlWXfN")
#project = rf.workspace("roboflow-100").project("road-signs-6ih4y")
#dataset = project.version(2).download("yolov8")


# Training the model using the Roboflow dataset
model = ultralytics.YOLO()
model = model.train(dataset_yaml="datasets/road_signs_dataset/data.yaml", imgsz=640, epochs=100, batch=16, model="yolov8s.pt", verbose=True, name="road_signs_dataset")