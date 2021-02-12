from __future__ import division
import numpy as np
import os
import cv2
import torch
from core.models import *
import pickle as pkl
from torch.autograd import Variable
import imageio
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

DRIVEDataSetPath = 'D:\\FrangiNet\\data\\DRIVE'
STAREDataSetPath = 'D:\\FrangiNet\\data\\Stare'
CHASEDB1DataSetPath = 'D:\\FrangiNet\\data\\CHASEDB1'

def get_data(dataset, img_name, img_size=256, gpu=True, flag='train'):

    def get_label(label):
        tmp_gt = label.copy()
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    imageGreys = []
    labels = []
    tmp_gts = []

    img_shape =[]
    label_ori = []
    Mask_ori =[]
    batch_size = len(img_name)

    for i in range(batch_size):
        if dataset == "DRIVE":
            img_path = os.path.join(DRIVEDataSetPath, 'images', img_name[i].rstrip('\n'))
            if flag == 'train':
                label_name = img_name[i].rstrip('\n')[:-12] + 'manual1.gif'
                mask_name = img_name[i].rstrip('\n')[:-12] + 'training_mask.gif'
            else:
                label_name = img_name[i].rstrip('\n')[:-8] + 'manual1.gif'
                mask_name = img_name[i].rstrip('\n')[:-8] + 'test_mask.gif'

            mask_path = os.path.join(DRIVEDataSetPath, 'masks', mask_name)
            FOVmask = imageio.mimread(mask_path)
            if FOVmask is not None:
                FOVmask = np.array(FOVmask)
                FOVmask = FOVmask[0]

            label_path = os.path.join(DRIVEDataSetPath, 'label', label_name)
            img = cv2.imread(img_path)
            label = imageio.mimread(label_path)
            if label is not None:
                label = np.array(label)
                label = label[0]

        if dataset == "STARE":
            img_path = os.path.join(STAREDataSetPath, 'images', img_name[i].rstrip('\n'))
            #label_name = img_name[i].rstrip('\n')[:-4] + '.vk.ppm'
            label_name = img_name[i].rstrip('\n')[:-4] + '.ah.ppm'
            label_path = os.path.join(STAREDataSetPath, 'labels', label_name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            if label is not None:
                label = label[:,:,0]
            mask_name = img_name[i].rstrip('\n')[:-4] + '.jpg'
            mask_path = os.path.join(STAREDataSetPath, 'Masks', mask_name)
            FOVmask = cv2.imread(mask_path)
            FOVmask = FOVmask[:, :, 0]

        if dataset == "CHASEDB1":
            img_path = os.path.join(CHASEDB1DataSetPath, 'images', img_name[i].rstrip('\n'))
            label_name = img_name[i].rstrip('\n')[:-4] + '_1stHO.png'
            label_path = os.path.join(CHASEDB1DataSetPath, 'label', label_name)
            mask_name = img_name[i].rstrip('\n')
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            if label is not None:
                label = label[:,:,0]
            mask_path = os.path.join(CHASEDB1DataSetPath, 'Masks', mask_name)
            FOVmask = cv2.imread(mask_path)
            FOVmask = FOVmask[:,:,0]
            #a = 1

        img_shape.append(img.shape)
        label_ori.append(label)
        label[label == 255] = 1

        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(FOVmask, (img_size, img_size), interpolation=cv2.INTER_AREA)

        segmap = SegmentationMapsOnImage(label, shape=label.shape)
        seq = iaa.Sequential([
            #iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(rotate=(-20, 20)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            #iaa.Flipud(0.2),
            #iaa.ElasticTransformation(alpha =50, sigma=5)  # apply water effect (affects segmaps)
            iaa.GammaContrast((0.5, 2.0))
            #iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
        ], random_order=True)

        #image_aug = rotate.augment_image(img)
        if flag == 'train':
            img, label = seq(image=img, segmentation_maps=segmap)
            label = np.squeeze(label.arr)
        #print("Augmented:")
        #ia.imshow(image_aug)
        #cv2.imshow('img', img)
        #cv2.imwrite('img.png', label.)
        #.imwrite('img2.png', segmaps_aug_i.arr*255)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGrey = imgGrey[np.newaxis,:,:]

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()
        imgGrey = Variable(torch.from_numpy(imgGrey)).double()

        if gpu:
            img = img.cuda()
            imgGrey = imgGrey.cuda()

        label, tmp_gt = get_label(label)
        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)
        imageGreys.append(imgGrey)
        Mask_ori.append(mask)

    images = torch.stack(images)
    imageGreys = torch.stack(imageGreys)

    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    if flag:
        label_ori = np.stack(label_ori)

    return images, imageGreys, labels, tmp_gts, img_shape, label_ori, Mask_ori

def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    return  meanIU,Acc,Se,Sp,IU

def get_model(model_name):
    if model_name=='M_Net':
        return M_Net
    if model_name=='UNet512':
        return UNet512
    if model_name=='SU_Net':
        return SU_Net
    if model_name=='UNet512_sideoutput':
        return UNet512_sideoutput

#dataset_list = ['DRIVE', "STARE", "CHASEDB1"]
def get_img_list(dataset, SubID, flag='train'):
    if dataset == "DRIVE":
        if flag=='train':
            with open(os.path.join(DRIVEDataSetPath, "DRIVEtraining.txt"),'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(DRIVEDataSetPath, "DRIVEtesting.txt"),'r') as f:
                img_list = f.readlines()

    if dataset == "STARE":
        TrainFileName = "Staretraining" + str(SubID) + '.txt'
        TestFileName = "Staretesting" + str(SubID) + '.txt'
        #TrainFileName = "StareTraining.txt"
        #TestFileName = "StareTesting.txt"
        #TrainFileName = "StareTrainingFold2.txt"
        #TestFileName = "StareTestingFold2.txt"
        if flag=='train':
            with open(os.path.join(STAREDataSetPath, TrainFileName), 'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(STAREDataSetPath, TestFileName), 'r') as f:
                img_list = f.readlines()

    if dataset == "CHASEDB1":
        if flag=='train':
            with open(os.path.join(CHASEDB1DataSetPath, "CHASEDB1training.txt"),'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(CHASEDB1DataSetPath, "CHASEDB1testing.txt"),'r') as f:
                img_list = f.readlines()

    return img_list