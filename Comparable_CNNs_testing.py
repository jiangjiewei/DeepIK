"""
DeepIK: an intelligent system to diagnose infectious keratitis using slit lamp photographs.
Here is the code for comparative CNNs.
Jiewei Jiang
10,20,2023

"""

import argparse
import os
import random
import shutil
import time
import warnings
import timm
import PIL
import cv2
from shutil import copyfile
import pickle
from PIL import Image
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from rep_VGGNET import *
from xception import xception
from repvgg import *
from repvgg_plus import *
from models import build_model
from inceptionResnet_V2 import InceptionResnetV2, inceptionresnetv2
from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./dataA_0802_jiang_8_2',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Transform_base',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate moinrtdel on validation set')
parser.add_argument('--pretrained', default='pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fine-tuning',default='True', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu',type=int,
                    help='GPU id to use.')  #default=3,
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def comparable_cnns_test():
    args = parser.parse_args()

    dataset_subdir = [
					  #   'test',
                      # 'external_test_2023_0718/1回顾性医院',
                      # 'external_test_2023_0718/兰州大学第二医院',
                      # 'external_test_2023_0718/南京医科大学附属眼科医院',
                      # 'external_test_2023_0718/南昌大学附属眼科医院',
                      # 'external_test_2023_0718/大连市第三人民医院',
                      # 'external_test_2023_0718/宁夏眼科医院',
                      # 'external_test_2023_0718/武汉大学人民医院',
                      # 'external_test_2023_0718/温州眼视光前瞻',
                      # 'external_test_2023_0718/湘雅三院',
                      # 'external_test_2023_0718/福建医科大学附属第一人民医院',
                      # 'external_test_2023_0718/西安市第一医院',
                      # 'external_test_2023_0718/贵州医院0716',
                        'external_test_2023_0718/1回顾性医院0805',
                        'external_test_2023_0718/宁波市眼科医院0805',
                     ]
    model_names = [
                'densenet121',
                # 'efficientnet-b7',
                # 'xception',
                # 'inception_v3',
                # 'RepVGG_B2g4',
                # 'Transform_large',
                'Transform_base',
                'inceptionresnetv2',
                # 'resnet50'
                #'alexnet'
                ]
    for index_name in range(0,len(model_names)):
        args.arch = model_names[index_name]
        for i in range(0,len(dataset_subdir)):
            dataset_dir = './dataA_0802_jiang_8_2/' + dataset_subdir[i]
            resultset_dir = './result_final_2023_jiang_8_2_0708_mt_tmp_test/' + dataset_subdir[i]
            args, model, val_transforms = load_modle_trained(args)
            mk_result_dir(args, resultset_dir)
            comparable_cnns_test_exec(args, model, val_transforms, dataset_dir,resultset_dir)



def load_modle_trained(args):

    normalize = transforms.Normalize(mean = [0.57135975, 0.33066592, 0.22625962], std = [0.20345019, 0.17539863, 0.16307473])
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    print("=> loading checkpoint### ", args.arch)
    if args.arch.find('alexnet') != -1:
        pre_name = './alexnet'
    elif args.arch.find('inception_v3') != -1:
        pre_name = './inception_v3'
    elif args.arch.find('densenet121') != -1:
        pre_name = './densenet121'
    elif args.arch.find('efficientnet-b0') != -1:
        pre_name = './efficientnet-b0'
    elif args.arch.find('efficientnet-b7') != -1:
        pre_name = './efficientnet-b7'
    elif args.arch.find('xception') != -1:
        pre_name = './xception'
    elif args.arch.find('inceptionresnetv2') != -1:
        pre_name = './inceptionresnetv2'
    elif args.arch.find('RepVGG_B2g4') != -1:
        pre_name = './RepVGG_B2g4'
    elif args.arch.find('Transform_large') != -1:
        pre_name = './Transform_large'
    elif args.arch.find('Transform_base') != -1:
        pre_name = './Transform_base'
    elif args.arch.find('resnet50') != -1:
        pre_name = './resnet50'
    else:
        print('### please check the args.arch###')
        exit(-1)
    PATH = pre_name + '_model_best.pth.tar'

    if args.arch.find('alexnet') != -1:
        model = models.__dict__[args.arch](pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 4)
    elif args.arch.find('inception_v3') != -1:
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        num_auxftrs = model.AuxLogits.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
        model.AuxLogits.fc = nn.Linear(num_auxftrs, 5)
        model.aux_logits = False
    elif args.arch.find('xception') != -1:
        model = xception(num_classes = 5)
    elif args.arch.find('densenet121') != -1:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 5)
    elif args.arch.find('efficientnet-b0') != -1:
        model = EfficientNet.from_pretrained('efficientnet-b0', advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 5)
    elif args.arch.find('efficientnet-b7') != -1:
        model = EfficientNet.from_pretrained('efficientnet-b7', advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 5)
    elif args.arch.find('RepVGG_B2g4') != -1:
        model = create_RepVGG_B2g4()
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, 5)
    elif args.arch.find('Transform_large') != -1:
        model = build_model('swin_large')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 5)
    elif args.arch.find('Transform_base') != -1:
        model = build_model('swin_base')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 5)
    elif args.arch.find('inceptionresnetv2') != -1:
        model = timm.create_model('inception_resnet_v2',pretrained=True)
        inchannel = model.classif.in_features
        model.classif = nn.Linear(inchannel, 5)
    elif args.arch.find('resnet') != -1:  # ResNet
        model = models.__dict__[args.arch](pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    else:
        print('### please check the args.arch for load model in testing###')
        exit(-1)

    print(model)
    if args.arch.find('alexnet') == -1:
        model = torch.nn.DataParallel(model).cuda()  #for modles trained by multi GPUs: densenet inception_v3 resnet50
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    if args.arch.find('alexnet') != -1:
        model = torch.nn.DataParallel(model).cuda()   #for models trained by single GPU: Alexnet
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    print('best_epoch and best_acc1 is: ' ,start_epoch   , best_acc1)
    return args, model, val_transforms

def mk_result_dir(args,testdata_dir='./data/val1'):
    testdatadir = testdata_dir
    model_name = args.arch
    result_dir = testdatadir + '/' + model_name
    grade1_grade2 = 'Amoeba_Bacteria'
    grade1_grade3 = 'Amoeba_Fungus'
    grade1_grade4 = 'Amoeba_Virus'
    grade1_grade5 = 'Amoeba_Others'

    grade2_grade1 = 'Bacteria_Amoeba'
    grade2_grade3 = 'Bacteria_Fungus'
    grade2_grade4 = 'Bacteria_Virus'
    grade2_grade5 = 'Bacteria_Others'

    grade3_grade1 = 'Fungus_Amoeba'
    grade3_grade2 = 'Fungus_Bacteria'
    grade3_grade4 = 'Fungus_Virus'
    grade3_grade5 = 'Fungus_Others'

    grade4_grade1 = "Virus_Amoeba"
    grade4_grade2 = "Virus_Bacteria"
    grade4_grade3 = "Virus_Fungus"
    grade4_grade5 = "Virus_Others"

    grade5_grade1 = "Others_Amoeba"
    grade5_grade2 = "Others_Bacteria"
    grade5_grade3 = "Others_Fungus"
    grade5_grade4 = "Others_Virus"


    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
        os.makedirs(result_dir + '/' + grade1_grade2)
        os.makedirs(result_dir + '/' + grade1_grade3)
        os.makedirs(result_dir + '/' + grade1_grade4)
        os.makedirs(result_dir + '/' + grade1_grade5)
        os.makedirs(result_dir + '/' + grade2_grade3)
        os.makedirs(result_dir + '/' + grade2_grade1)
        os.makedirs(result_dir + '/' + grade2_grade4)
        os.makedirs(result_dir + '/' + grade2_grade5)
        os.makedirs(result_dir + '/' + grade3_grade1)
        os.makedirs(result_dir + '/' + grade3_grade2)
        os.makedirs(result_dir + '/' + grade3_grade4)
        os.makedirs(result_dir + '/' + grade3_grade5)
        os.makedirs(result_dir + '/' + grade4_grade1)
        os.makedirs(result_dir + '/' + grade4_grade2)
        os.makedirs(result_dir + '/' + grade4_grade3)
        os.makedirs(result_dir + '/' + grade4_grade5)
        os.makedirs(result_dir + '/' + grade5_grade1)
        os.makedirs(result_dir + '/' + grade5_grade2)
        os.makedirs(result_dir + '/' + grade5_grade3)
        os.makedirs(result_dir + '/' + grade5_grade4)

        # os.makedirs(result_dir + '/' + grade1_grade1)
        # os.makedirs(result_dir + '/' + grade2_grade2)
        # os.makedirs(result_dir + '/' + grade3_grade3)
        # os.makedirs(result_dir + '/' + grade4_grade4)
        # os.makedirs(result_dir + '/' + grade5_grade5)

def comparable_cnns_test_exec(args,model,val_transforms,testdata_dir='./data/val1',resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg','.JPEG','.BMP','.tif','.TIF','.png','.PNG')
    grade1_grade2 = 'Amoeba_Bacteria'
    grade1_grade3 = 'Amoeba_Fungus'
    grade1_grade4 = 'Amoeba_Virus'
    grade1_grade5 = 'Amoeba_Others'

    grade2_grade1 = 'Bacteria_Amoeba'
    grade2_grade3 = 'Bacteria_Fungus'
    grade2_grade4 = 'Bacteria_Virus'
    grade2_grade5 = 'Bacteria_Others'

    grade3_grade1 = 'Fungus_Amoeba'
    grade3_grade2 = 'Fungus_Bacteria'
    grade3_grade4 = 'Fungus_Virus'
    grade3_grade5 = 'Fungus_Others'

    grade4_grade1 = "Virus_Amoeba"
    grade4_grade2 = "Virus_Bacteria"
    grade4_grade3 = "Virus_Fungus"
    grade4_grade5 = "Virus_Others"

    grade5_grade1 = "Others_Amoeba"
    grade5_grade2 = "Others_Bacteria"
    grade5_grade3 = "Others_Fungus"
    grade5_grade4 = "Others_Virus"

    grade1_grade1 = 'Amoeba_Amoeba'
    grade2_grade2 = 'Bacteria_Bacteria'
    grade3_grade3 = 'Fungus_Fungus'
    grade4_grade4 = 'Virus_Virus'
    grade5_grade5 = 'Others_Others'


    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        grade1_grade4_num = 0
        grade1_grade5_num = 0
        list_grade1_grade2=[grade1_grade2]
        list_grade1_grade3=[grade1_grade3]
        list_grade1_grade4=[grade1_grade4]
        list_grade1_grade5=[grade1_grade5]
        grade1_2=[grade1_grade2]
        grade1_3=[grade1_grade3]
        grade1_1=[grade1_grade1]
        grade1_4=[grade1_grade4]
        grade1_5=[grade1_grade5]
        root = testdatadir + '/Amoeba'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ",len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
			
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)

            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            # print(prob_list)
            # print(pred_0)

            grade1_num = grade1_num + 1
            # print(grade1_num)
            if pred_0 == 0:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
            elif pred_0 == 1:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/Amoeba_Bacteria' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                # print('ok to location')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/Amoeba_Fungus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                # print('ok to location')
                grade1_grade5_num = grade1_grade5_num + 1
                list_grade1_grade5.append(img)
                grade1_5.append(prob_list)
                file_new_1 = desdatadir + '/Amoeba_Others' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade4_num = grade1_grade4_num + 1
                list_grade1_grade4.append(img)
                grade1_4.append(prob_list)
                file_new_1 = desdatadir + '/Amoeba_Virus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num,grade1_grade4_num,grade1_grade5_num)
        # sen_1 = grade1_grade1_num/(grade1_grade1_num+grade1_grade2_num+grade1_grade3_num+grade1_grade4_num+grade1_grade5_num)
        # sum_1 = grade1_grade1_num+grade1_grade2_num+grade1_grade3_num+grade1_grade4_num+grade1_grade5_num

        grade2_grade1_num = 0
        grade2_grade3_num = 0
        grade2_grade2_num = 0
        grade2_grade4_num = 0
        grade2_grade5_num = 0
        grade2_num = 0
        list_grade2_grade1=[grade2_grade1]
        list_grade2_grade3=[grade2_grade3]
        list_grade2_grade4=[grade2_grade4]
        list_grade2_grade5=[grade2_grade5]
        grade2_1=[grade2_grade1]
        grade2_3=[grade2_grade3]
        grade2_2=[grade2_grade2]
        grade2_4=[grade2_grade4]
        grade2_5=[grade2_grade5]
        root = testdatadir + '/Bacteria'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            # img_PIL.show()  # 原始图片
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            # image = cv2.imread(os.path.join(root, img))  # image = image.unsqueeze(0) # PIL_image = Image.fromarray(image)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade2_num = grade2_num + 1
            if pred_0 == 0:
                # print('location to ok')
                grade2_grade1_num = grade2_grade1_num + 1
                list_grade2_grade1.append(img)
                grade2_1.append(prob_list)
                file_new_1 = desdatadir + '/Bacteria_Amoeba' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                # print('location to location')
                grade2_grade2_num = grade2_grade2_num + 1
                grade2_2.append(prob_list)

            elif pred_0 == 2:
                # print('location to quality')
                grade2_grade3_num = grade2_grade3_num + 1
                list_grade2_grade3.append(img)
                grade2_3.append(prob_list)
                file_new_1 = desdatadir + '/Bacteria_Fungus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                # print('location to quality')
                grade2_grade4_num = grade2_grade4_num + 1
                list_grade2_grade4.append(img)
                grade2_4.append(prob_list)
                file_new_1 = desdatadir + '/Bacteria_Virus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade2_grade5_num = grade2_grade5_num + 1
                list_grade2_grade5.append(img)
                grade2_5.append(prob_list)
                file_new_1 = desdatadir + '/Bacteria_Others' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade2_grade1_num, grade2_grade2_num, grade2_grade3_num, grade2_grade4_num, grade2_grade5_num)
        # sen_2 = grade2_grade2_num/(grade2_grade1_num+grade2_grade2_num+grade2_grade3_num+grade2_grade4_num+grade2_grade5_num)
        # sum_2 = grade2_grade1_num+grade2_grade2_num+grade2_grade3_num+grade2_grade4_num+grade2_grade5_num

        grade3_grade1_num = 0
        grade3_grade3_num = 0
        grade3_grade2_num = 0
        grade3_grade4_num = 0
        grade3_grade5_num = 0
        grade3_num = 0
        list_grade3_grade1=[grade3_grade1]
        list_grade3_grade4=[grade3_grade4]
        list_grade3_grade2=[grade3_grade2]
        list_grade3_grade5=[grade3_grade5]
        grade3_1=[grade3_grade1]
        grade3_2=[grade3_grade2]
        grade3_3=[grade3_grade3]
        grade3_4=[grade3_grade4]
        grade3_5=[grade3_grade5]
        root = testdatadir + '/Fungus'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade3_num = grade3_num + 1
            if pred_0 == 0:
                grade3_grade1_num = grade3_grade1_num + 1
                list_grade3_grade1.append(img)
                grade3_1.append(prob_list)
                file_new_1 = desdatadir + '/Fungus_Amoeba' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade3_grade2_num = grade3_grade2_num + 1
                list_grade3_grade2.append(img)
                grade3_2.append(prob_list)
                file_new_1 = desdatadir + '/Fungus_Bacteria' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade3_grade3_num = grade3_grade3_num + 1
                grade3_3.append(prob_list)
            elif pred_0 == 4:
                grade3_grade4_num = grade3_grade4_num + 1
                list_grade3_grade4.append(img)
                grade3_4.append(prob_list)
                file_new_1 = desdatadir + '/Fungus_Virus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade3_grade5_num = grade3_grade5_num + 1
                list_grade3_grade5.append(img)
                grade3_5.append(prob_list)
                file_new_1 = desdatadir + '/Fungus_Others' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade3_grade1_num, grade3_grade2_num, grade3_grade3_num, grade3_grade4_num, grade3_grade5_num)
        # sen_3 = grade3_grade3_num/(grade3_grade1_num+ grade3_grade2_num+  grade3_grade3_num+  grade3_grade4_num+  grade3_grade5_num)
        # sum_3 = grade3_grade1_num+ grade3_grade2_num+  grade3_grade3_num+  grade3_grade4_num+  grade3_grade5_num

        grade4_grade1_num = 0
        grade4_grade3_num = 0
        grade4_grade2_num = 0
        grade4_grade4_num = 0
        grade4_grade5_num = 0
        grade4_num = 0
        list_grade4_grade1 = [grade4_grade1]
        list_grade4_grade3 = [grade4_grade3]
        list_grade4_grade2 = [grade4_grade2]
        list_grade4_grade5 = [grade4_grade5]
        grade4_1 = [grade4_grade1]
        grade4_2 = [grade4_grade2]
        grade4_3 = [grade4_grade3]
        grade4_4 = [grade4_grade4]
        grade4_5 = [grade4_grade5]
        root = testdatadir + '/Virus'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade4_num = grade4_num + 1
            if pred_0 == 0:
                grade4_grade1_num = grade4_grade1_num + 1
                list_grade4_grade1.append(img)
                grade4_1.append(prob_list)
                file_new_1 = desdatadir + '/Virus_Amoeba' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade4_grade2_num = grade4_grade2_num + 1
                list_grade4_grade2.append(img)
                grade4_2.append(prob_list)
                file_new_1 = desdatadir + '/Virus_Bacteria' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade4_grade3_num = grade4_grade3_num + 1
                list_grade4_grade3.append(img)
                grade4_3.append(prob_list)
                file_new_1 = desdatadir + '/Virus_Fungus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                grade4_grade4_num = grade4_grade4_num + 1
                grade4_4.append(prob_list)
            else:
                grade4_grade5_num = grade4_grade5_num + 1
                list_grade4_grade5.append(img)
                grade4_5.append(prob_list)
                file_new_1 = desdatadir + '/Virus_Others' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num, grade4_grade5_num)
        # sen_4 = grade4_grade4_num/(grade4_grade1_num+ grade4_grade2_num+  grade4_grade3_num+  grade4_grade4_num+  grade4_grade5_num)
        # sum_4 = grade4_grade1_num+ grade4_grade2_num+  grade4_grade3_num+  grade4_grade4_num+  grade4_grade5_num

        grade5_grade1_num = 0
        grade5_grade3_num = 0
        grade5_grade2_num = 0
        grade5_grade4_num = 0
        grade5_grade5_num = 0
        grade5_num = 0
        list_grade5_grade1 = [grade5_grade1]
        list_grade5_grade3 = [grade5_grade3]
        list_grade5_grade2 = [grade5_grade2]
        list_grade5_grade4 = [grade5_grade4]
        grade5_1 = [grade5_grade1]
        grade5_2 = [grade5_grade2]
        grade5_3 = [grade5_grade3]
        grade5_4 = [grade5_grade4]
        grade5_5 = [grade5_grade5]
        root = testdatadir + '/Others'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade5_num = grade5_num + 1
            if pred_0 == 0:
                grade5_grade1_num = grade5_grade1_num + 1
                list_grade5_grade1.append(img)
                grade5_1.append(prob_list)
                file_new_1 = desdatadir + '/Others_Amoeba' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade5_grade2_num = grade5_grade2_num + 1
                list_grade5_grade2.append(img)
                grade5_2.append(prob_list)
                file_new_1 = desdatadir + '/Others_Bacteria' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade5_grade3_num = grade5_grade3_num + 1
                list_grade5_grade3.append(img)
                grade5_3.append(prob_list)
                file_new_1 = desdatadir + '/Others_Fungus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                grade5_grade5_num = grade5_grade5_num + 1
                grade5_5.append(prob_list)
            else:
                grade5_grade4_num = grade5_grade4_num + 1
                list_grade5_grade4.append(img)
                grade5_4.append(prob_list)
                file_new_1 = desdatadir + '/Others_Virus' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade5_grade1_num, grade5_grade2_num, grade5_grade3_num, grade5_grade4_num, grade5_grade5_num)
        # sen_5 = grade5_grade5_num/(grade5_grade1_num+ grade5_grade2_num+  grade5_grade3_num+  grade5_grade4_num+  grade5_grade5_num)
        # sum_5 = grade5_grade1_num+ grade5_grade2_num+  grade5_grade3_num+  grade5_grade4_num+  grade5_grade5_num
        # acc_sum =  grade1_grade1_num + grade2_grade2_num +grade3_grade3_num + grade4_grade4_num +  grade5_grade5_num
        # acc = acc_sum/(sum_1+sum_2+sum_3+sum_4+sum_5)
    confusion_matrix = [ [grade1_grade1_num, grade1_grade2_num, grade1_grade3_num, grade1_grade4_num, grade1_grade5_num],
                         [grade2_grade1_num, grade2_grade2_num, grade2_grade3_num, grade2_grade4_num, grade2_grade5_num],
                         [grade3_grade1_num, grade3_grade2_num, grade3_grade3_num, grade3_grade4_num, grade3_grade5_num],
                         [grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num, grade4_grade5_num],
                         [grade5_grade1_num, grade5_grade2_num, grade5_grade3_num, grade5_grade4_num, grade5_grade5_num]]
    # print('sen 1,2,3,4,5 and acc are:', sen_1,sen_2,sen_3,sen_4,sen_5,acc)

    print('confusion_matrix:')
    print (confusion_matrix)

    result_confusion_file = args.arch + '_1.txt'
    result_pro_file =  args.arch + '_2.txt'
    result_value_bin = args.arch + '_3.txt'


    with open(desdatadir + '/' + result_confusion_file, "w") as file_object:
        for i in confusion_matrix:
            file_object.writelines(str(i) + '\n')
        file_object.writelines('ERROR_images\n')
        for i in list_grade1_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade5:
            file_object.writelines(str(i) + '\n')

        for i in list_grade2_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade5:
            file_object.writelines(str(i) + '\n')

        for i in list_grade3_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade5:
            file_object.writelines(str(i) + '\n')

        for i in list_grade4_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade5:
            file_object.writelines(str(i) + '\n')

        for i in list_grade5_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade4:
            file_object.writelines(str(i) + '\n')
        file_object.close()

    with open(desdatadir + '/' + result_pro_file, "w") as file_object:
        for i in grade1_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')


        for i in grade2_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade3_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade4_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade5_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        file_object.close()

    with open(desdatadir + '/' + result_value_bin, "wb") as file_object:
        pickle.dump(confusion_matrix, file_object)  # 顺序存入变量
        pickle.dump(grade1_1, file_object)
        pickle.dump(grade1_2, file_object)
        pickle.dump(grade1_3, file_object)
        pickle.dump(grade1_4, file_object)
        pickle.dump(grade1_5, file_object)
        pickle.dump(grade2_1, file_object)
        pickle.dump(grade2_2, file_object)
        pickle.dump(grade2_3, file_object)
        pickle.dump(grade2_4, file_object)
        pickle.dump(grade2_5, file_object)
        pickle.dump(grade3_1, file_object)
        pickle.dump(grade3_2, file_object)
        pickle.dump(grade3_3, file_object)
        pickle.dump(grade3_4, file_object)
        pickle.dump(grade3_5, file_object)
        pickle.dump(grade4_1, file_object)
        pickle.dump(grade4_2, file_object)
        pickle.dump(grade4_3, file_object)
        pickle.dump(grade4_4, file_object)
        pickle.dump(grade4_5, file_object)
        pickle.dump(grade5_1, file_object)
        pickle.dump(grade5_2, file_object)
        pickle.dump(grade5_3, file_object)
        pickle.dump(grade5_4, file_object)
        pickle.dump(grade5_5, file_object)
        file_object.close()


if __name__ == '__main__':

    comparable_cnns_test()

