from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision import transforms

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2, torch

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std
    def __call__(self,tensor):
        for t,m,s in zip(tensor,self.mean,self.std):
            t.mul_(s).add_(m)
        return tensor

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        # 
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w))) # [1X512].dot([512,h*w(4X4)]) 하나의 클래스에 대해서
        cam = cam.reshape(h, w) # [1X16] -> [4X4]
        cam = cam - np.min(cam) # 가장 작은 값을 0으로 정규화 ( CAM으로 표시시 색깔 표현을 위한 )
        cam_img = cam / np.max(cam) # max로 나누는 경우 최대값은 1로 정규화
        cam_img = np.uint8(255 * cam_img) # 0~1사이의 값을 0~255값으로 정규화
        
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam_CIFAR10(net,data,pred, features_blobs, classes,file_index):
    
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) # parameter fully connected layer input =  512 output = 10
    # weight_softmax.shape = (10,512) 10 = output_channel 512 = input_channel
    '''
    for idx,params_print in enumerate(params):
        print(idx,":",np.array(params_print.data.cpu()).shape)
    '''

    logit = pred

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    unorm = UnNormalize()
    data = unorm(data)
    data = data.cpu().numpy()
    data = np.squeeze(data,axis=0)
    data = data.transpose(1,2,0)
    # 0 보다 작은 값은 전부 0으로
    data[data<0.] = 0.
    # 1 보다 큰 값은 전부 1로
    data[data>1.] = 1.
    img = data
    # os.makedirs('./checkCAM/',exist_ok=True)
    # plt.imsave('./checkCAM/origin%d.jpg'%file_index, data)

    # img = cv2.imread('./checkCAM/origin%d.jpg'%file_index)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height)) # [4X4] -> [128X128]
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET) # cv2.COLORMAP_JET => 0 : 파란색 255 : 빨간색
    # result = cv2.resize(result,(224,224))
    # cv2.imwrite('./checkCAM/result%d.jpg'%file_index, result)
    return img,heatmap
    # img = cv2.imread('./checkCAM/origin%d.jpg'%file_index)
    # img = cv2.resize(img,(224,224))
    # cv2.imwrite('./checkCAM/origin%d.jpg'%file_index,img)
