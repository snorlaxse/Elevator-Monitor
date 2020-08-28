import cv2
import torch
from model import LSCCNN
from matplotlib import pyplot as plt
import numpy as np

def detect_image(src_image):
        
    image = cv2.imread(src_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred_dot_map, pred_box_map, img_out = network.predict_single_image(image, nms_thresh=0.10)

    head_idx = np.where(pred_dot_map > 0)
    Y, X = head_idx[-2] , head_idx[-1]
    predict_head_num = len(X)
    print(predict_head_num)
    return predict_head_num

if __name__ == "__main__":

    checkpoint_path = './weights/part_a_scale_4_epoch_13_weights.pth'

    network = LSCCNN(checkpoint_path=checkpoint_path)
    if torch.cuda.is_available():
        network.cuda()
    network.eval()

    src_image = './source/hands.jpg'
    detect_image(src_image)
