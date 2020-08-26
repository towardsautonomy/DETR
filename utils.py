import torchvision.transforms as T
import numpy as np
import cv2

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Minimum confidence for detections
conf_thres = 0.7

# This function is used to draw bounding-box
def draw_bbox(img_bgr, pred):
    # create a copy of image for bounding-box visualization
    img_copy = img_bgr.copy()
    
    # Extract class probability and bounding-box
    pred_logits = pred['pred_logits'][0]
    pred_boxes = pred['pred_boxes'][0]
    
    # get softmax of logits and choose the index with highest probability
    pred_prob = pred_logits.softmax(-1)
    pred_prob_np = pred_prob.cpu().numpy()
    pred_idx = np.argmax(pred_prob_np, -1)
    
    ## Filter out detections and draw bounding-box
    # iterate through predictions
    for i, (idx, box) in enumerate(zip(pred_idx, pred_boxes)):
        if (idx >= len(CLASSES)) or (pred_prob_np[i][idx] < conf_thres):
            continue

        label = CLASSES[idx]
        box = box.cpu().numpy() * [img_bgr.shape[1], img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[0]]
        x, y, w, h = box

        # get bbox corners
        x0, y0, x1, y1 = int(x-(w/2)), int(y-(h/2)), int(x+(w/2)), int(y+(h/2))

        # draw bounding-box
        cv2.rectangle(img_copy, (x0,y0), (x1,y1), thickness=2, color=(0,0,255))
        
        # A rectangle behind text
        cv2.rectangle(img_copy, (x0,y0-20), (x0+100,y0), color=(0,0,0), thickness=-1)
        
        # write object id
        cv2.putText(img_copy, '{} {:.1f}%'.format(label, pred_prob_np[i][idx]*100.0), 
            (x0+5, y0-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4,
            color=(255,255,255),
            thickness=1,
            lineType=2)
        
    # return the image with bounding-box
    return img_copy