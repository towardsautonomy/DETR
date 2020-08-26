# import packages
import cv2
import glob
import torch
import numpy as np
import time
from utils import *

# image path
# img_path = '/media/shubham/GoldMine/datasets/KITTI/raw/2011_09_26/*/image_02/data/*.png'
img_path ='/media/shubham/GoldMine/datasets/nuScenes/v1.0-mini/samples/CAM_FRONT/*.jpg'

# write to video
GENERATE_VID = False
VID_FILENAME = './media/nuscenes_mini.mp4'
fps = 30
video = None

# Get model from PyTorch hub and load it into the GPU
detr_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
detr_model.eval()
detr_model = detr_model.cuda()

# get list of images
img_fnames = sorted(glob.glob(img_path))

# number of samples to try this on
n_samples = 1000

# list of time taken in milliseconds for each frame
time_taken = []

# iterate through all the images
for img_fname in img_fnames[:n_samples]:
    # read the image
    img_bgr = cv2.imread(img_fname)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # create video writer
    if (GENERATE_VID == True) and (img_fname is img_fnames[0]):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(VID_FILENAME,fourcc, fps, (img_bgr.shape[1], img_bgr.shape[0]))
        
    # convert to tensor and load to GPU
    img_tensor = transform(img_rgb).unsqueeze(0).cuda()
    
    # perform inference
    pred = None
    with torch.no_grad():
        # log start time
        t_start_ms = time.time_ns() / 100000.0
        # forward pass through model
        pred = detr_model(img_tensor)
        # log end time
        t_end_ms = time.time_ns() / 100000.0
        time_taken.append(t_end_ms-t_start_ms)
        
    img_bbox = draw_bbox(img_bgr, pred)
    
    # write to video
    if GENERATE_VID == True:
        video.write(img_bbox)
        
    # visualize image with bounding-box
    cv2.imshow('img', img_bbox)
    cv2.waitKey(1)
    
avg_time_taken = np.mean(time_taken)
print('Tested on {} frames. Average time taken for prediction: {:.2f} ms; FPS: {:.2f}'.format(min(n_samples, len(img_fnames)),  
                                                                                              avg_time_taken, 
                                                                                              1000.0/avg_time_taken))

# close all open windows
cv2.destroyAllWindows()
if GENERATE_VID == True:
    video.release()