import torch
import cv2
import numpy as np
from torch.nn import functional as F

    
class ModelOutputs_resnet():
    def __init__(self, model, target_layers, target_sub_layers):
        self.model = model
        self.target_layers = target_layers
        self.target_sub_layers = target_sub_layers
        self.gradients = []
        self.target_feature_maps = []
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        self.gradients = []
        for name, module in self.model.named_children(): # 모든 layer에 대해서 직접 접근
            x = module(x)
            if name== 'avgpool': # avgpool이후 fully connect하기 전 data shape을 flatten시킴
                x = torch.flatten(x,1)
            # if name in self.target_layers: # target_layer라면 해당 layer에서의 gradient를 저장
            #     for sub_name, sub_module in module[len(module)-1].named_children():
            #         if sub_name in self.target_sub_layers:
            #             x.register_hook(self.save_gradient) #
            #             target_feature_maps = x # x's shape = 512X14X14(C,W,H) feature map
            if name in ['layer1','layer2','layer3','layer4']: # target_layer라면 해당 layer에서의 gradient를 저장
                for sub_name, sub_module in module[len(module)-1].named_children():
                    if sub_name in self.target_sub_layers:
                        x.register_hook(self.save_gradient) #
                        self.target_feature_maps.append(x)
                        target_feature_maps = x
        return target_feature_maps, x # target_activation : target_activation_layer's feature maps // output : classification ( ImageNet's classes : 1000 )


class GradCam_resnet:
    def __init__(self, model, target_layer_names,target_sub_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:  # GPU일 경우 model을 cuda로 설정
            self.model = model.cuda()
        self.target_layer_names = target_layer_names
        self.extractor = ModelOutputs_resnet(self.model, target_layer_names,target_sub_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:  # GPU일 경우 input을 cuda로 변환하여 전달
            features, output = self.extractor(input.cuda()) 
        else:
            features, output = self.extractor(input)
        # print(features.shape,output.shape) # torch.Size([1, 512, 7, 7]) torch.Size([1, 1000])
        probs,idx = 0, 0
        if index == None:
            index = np.argmax(output.cpu().data.numpy())  # index = 정답이라고 추측한 class index
            h_x = F.softmax(output,dim=1).data.squeeze()
            probs, idx = h_x.sort(0,True)


        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  # 정답이라고 생각하는 class의 index 리스트 위치의 값만 1로
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)  # numpy배열을 tensor로 변환
        # requires_grad == True 텐서의 모든 연산에 대하여 추적
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)# loss에 대한 backprop대신 해당 one-hot에 대해서 back-prop수행(grad얻기 위해)

        # CAM
        params = list(self.model.parameters())
        
        return self.extractor.target_feature_maps, self.extractor.gradients, params , index, probs, idx

def make_heatmap_gradCAM(target,gradient):
    target = target.cpu().data.numpy() # feature(forward) 
    gradient = np.mean(gradient.cpu().data.numpy(),axis=(2,3))[0,:]
    bz, nc, h,w = target.shape
    target = target[0,:] # remove batch dim
    
    # Grad-CAM
    grad_cam = np.zeros(target.shape[1:], dtype=np.float32)  # 14X14

    for i, w in enumerate(gradient): # calcul grad_cam
        grad_cam += w * target[i, :, :]  # linear combination L^c_{Grad-CAM}에 해당하는 식에서 ReLU를 제외한 식

    grad_cam = np.maximum(grad_cam, 0)  # 0보다 작은 값을 제거
    grad_cam = cv2.resize(grad_cam, (224, 224))  # 224X224크기로 변환
    grad_cam = grad_cam - np.min(grad_cam)  #
    grad_cam = grad_cam / np.max(grad_cam)  # 위의 것과 해당 줄의 것은 0~1사이의 값으로 정규화하기 위한 정리
    
    return grad_cam

def make_heatmap_CAM(features,params,index):
    
    n,c,h,w = features.shape
    features = features.detach().cpu().numpy()
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    cam = weight_softmax[index].dot(features.reshape((n*c,h*w)))
    cam = cam.reshape(h,w)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))  # 224X224크기로 변환
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    return cam