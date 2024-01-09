"""
DeepIK: an intelligent system to diagnose infectious keratitis using slit lamp photographs.
Jiewei Jiang
10,20,2023

"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import cv2
from shutil import copyfile
import pickle
from PIL import Image
import torch.nn.functional as F
from inceptionResnet_V2 import InceptionResnetV2, inceptionresnetv2
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
from xception import xception
from rep_VGGNET import *
from repvgg import *
from repvgg_plus import *
from models import build_model
from ASL_loss import ASLSingleLabel
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

from focalloss import *
from densenet import densenet121
import sys
from efficientnet_pytorch import EfficientNet


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def _find_classes(self, dir):
	    # Adding comments by Jiang Jiewei on January 9, 2024 
		# The CustomImageFolder class is defined to specify the actual category labels produced by a dual-layer classifier of DeepIK model.
        # The first layer is a binary classifier, and the second layer is a five-class classifier. 
        class_to_idx={"Amoeba":[0,0],"Bacteria":[1,0],"Fungus":[2,0],"Others":[3,1],"Virus":[4,0]}
        # class_to_idx={"Amoeba":[0,0],"Bacteria":[1,1],"Fungus":[2,1],"Others":[3,2],"Virus":[4,3]}
        classes=["Amoeba","Bacteria","Fungus","Others","Virus"]
        return classes, class_to_idx

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./dataA_0802_jiang_8_2/',
                    help='path to dataset')

# Adding comments by Jiang Jiewei on January 9, 2024 
# Adding two-layer classifiers on the densenet121 network to train the DeepIK model, with the input parameter "densenet121" unchanged.
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
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
                    help='transfer learning + fine tuning - datasetAll only the last FC layer.')
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
best_loss = 1000


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_loss
    args.gpu = gpu

    pre_model_keritits = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            if args.arch.find('alexnet') != -1:
                model = models.__dict__[args.arch](pretrained=True)
            elif args.arch.find('inception_v3') != -1:
                model = models.inception_v3(pretrained=True)
            elif args.arch.find('densenet121') != -1:
                model = densenet121(pretrained=True)
            elif args.arch.find('densenet201') != -1:
                model = models.densenet201(pretrained=True)
            elif args.arch.find('inceptionresnet') != -1:
                model = inceptionresnetv2()
            elif args.arch.find('xception') != -1:
                model = xception(True)
            elif args.arch.find('RepVGG_A1') != -1:
                model = RepVGG_A1()
                model.load_state_dict(torch.load("./pretrained_model/RepVGG-A1-train.pth"))
            elif args.arch.find('RepVGG_B2g4') != -1:
                model = create_RepVGG_B2g4()
                model.load_state_dict(torch.load("./pretrained_model/RepVGG-B2g4-train.pth"))
            elif args.arch.find('Transform_large') != -1:
                model = build_model('swin_large')
                model.load_state_dict(torch.load("./pretrained_model/swin_large_patch4_window7_224_22kto1k.pth")["model"])
            elif args.arch.find('Transform_base') != -1:
                model = build_model("swin_base")
                model.load_state_dict(torch.load("./pretrained_model/swin_base_patch4_window7_224.pth")["model"])
            elif args.arch.find('resnet') != -1:  # ResNet
                model = models.__dict__[args.arch](pretrained=True)
            else:
                print('### please check the args.arch for load model in training###')
                exit(-1)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (datasetAll  only the last FC layer)")
        # Freeze Previous Layers(now we are using them as features extractor)
        #jiangjiewei
        # for param in model.parameters():
        #    param.requires_grad = False

        # Fine Tuning the last Layer For the new task
        # juge network: alexnet, inception_v3, densennet, resnet50
        if args.arch.find('alexnet') != -1:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 5)
        elif args.arch.find('inception_v3') != -1:
            num_ftrs = model.fc.in_features
            num_auxftrs = model.AuxLogits.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
            model.AuxLogits.fc =nn.Linear(num_auxftrs,5)
            model.aux_logits = False
        elif args.arch.find('densenet121') != -1:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 3)
        elif args.arch.find('densenet201') != -1:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 4)
        elif args.arch.find('xception') != -1:
            inchannel = model.fc.in_features
            model.fc = nn.Linear(inchannel, 5)
        elif args.arch.find('inceptionresnetv2') != -1:
            inchannel = model.last_linear.in_features
            model.last_linear = nn.Linear(inchannel, 4)
        elif args.arch.find('RepVGG_A1') != -1:
            inchannel = model.linear.in_features
            model.linear = nn.Linear(inchannel, 4)
        elif args.arch.find('RepVGG_B2g4') != -1:
            inchannel = model.linear.in_features
            model.linear = nn.Linear(inchannel, 5)
        elif args.arch.find('Transform_large') !=-1 or args.arch.find('Transform_base') !=-1:
            inchannel = model.head.in_features
            model.head = nn.Linear(inchannel, 5)
        elif args.arch.find('resnet') != -1: # ResNet
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
        else:
            print("###Error: Fine-tuning is not supported on this architecture.###")
            exit(-1)
        if args.arch.find('densenet121') != -1:
            checkpoint = torch.load("./pretrained_model/densenet121_model_best.pth.tar")
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            num_ftrs = model.classifier.in_features
			# Adding comments by Jiang Jiewei on January 9, 2024 
			# Setting the number of categories for the dual-layer classifier: the first-layer classifier has 2 categories, and the second-layer classifier has 5 categories.
            model.classifier = nn.Linear(num_ftrs, 2)
            model.classifier2 = nn.Linear(num_ftrs, 5)
            # model.classifier = nn.Linear(num_ftrs, 5)
            print('best_epoch and best_acc1 is: ', checkpoint['epoch'], checkpoint['best_acc1'])
        elif args.arch.find('Transform_base') != -1:
            # model = build_model("swin_base")
            inchannel = model.head.in_features
            model.head = nn.Linear(inchannel, 3)
            model = torch.nn.DataParallel(model).cuda()
            # model.load_state_dict(torch.load("./pretrained_model/swin_base_patch4_window7_224.pth")["model"])
            checkpoint = torch.load("./pretrained_model/Transform_base_model_best.pth.tar")
            # checkpoint = torch.load("./pretrained_model/Transform_base_model_best.pth.tar")['state_dict']
            # new_state_dict = OrderedDict()
            # for k,v in checkpoint.items():
            #     name = k[7:]
            #     new_state_dict[name] = v
            # model.load_state_dict(checkpoint['new_state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            inchannel = model.module.head.in_features
            model.module.head = nn.Linear(inchannel, 5)
            pre_model_keritits = 1

            print('best_epoch and best_acc1 is: ', checkpoint['epoch'], checkpoint['best_acc1'])
        print(model)
    else:
        parameters = model.parameters()
    # name, parma_1 = model.classifier[6].parameters()

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet-1') or args.arch.startswith('vgg-1'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        elif pre_model_keritits == 1:
            print(pre_model_keritits)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	# Adding comments by Jiang Jiewei on January 9, 2024 
    # In the loss function of the second layer, the weight for each category is set to address the issue of imbalanced data. 
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = nn.CrossEntropyLoss(weight=torch.Tensor([3, 5, 3, 1, 1])).cuda(args.gpu)# new
    # use_cuda = True
    # device = torch.device("cuda" if use_cuda else "cpu")
    # class_weights = torch.FloatTensor([1.0, 0.2, 1.0]).cuda()
    # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)


    if args.arch.find('alexnet') != -1:
        fine_tune_parameters =model.classifier[6].parameters()
    elif args.arch.find('inception_v3') != -1:
        fine_tune_parameters = model.module.fc.parameters()
    elif args.arch.find('densenet121') != -1:
        fine_tune_parameters = model.module.classifier.parameters()
    elif args.arch.find('densenet201') != -1:
        fine_tune_parameters = model.module.classifier.parameters()
    elif args.arch.find('xception') != -1:
        fine_tune_parameters = model.module.fc.parameters()
    elif args.arch.find('inceptionresnetv2') != -1:
        fine_tune_parameters = model.module.last_linear.parameters()
    elif args.arch.find('RepVGG_A1') != -1:
        fine_tune_parameters = model.module.linear.parameters()
    elif args.arch.find('RepVGG_B2g4') != -1:
        fine_tune_parameters = model.module.linear.parameters()
    elif (args.arch.find('Transform_large') != -1) or args.arch.find('Transform_base') != -1:
        fine_tune_parameters = model.module.head.parameters()

    elif args.arch.find('resnet') != -1:  # ResNet
        fine_tune_parameters = model.module.fc.parameters()
    else:
        print('### please check the ignored params ###')
        exit(-1)

    ignored_params = list(map(id, fine_tune_parameters))

    if args.arch.find('alexnet') != -1:
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())
    else:
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.module.parameters())

    # optimizer = torch.optim.SGD([{'params': base_params},  #model.parameters()
    #                             {'params': fine_tune_parameters, 'lr': 10*args.lr}],
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
	
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean = [0.57135975, 0.33066592, 0.22625962], std = [0.20345019, 0.17539863, 0.16307473])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    train_dataset = CustomImageFolder(
        traindir,
        transforms.Compose([
            # transforms.Resize((256, 256), interpolation=PIL.Image.BICUBIC),
            # transforms.Resize((224, 224)),
            transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
            # transforms.RandomResizedCrop((image_size, image_size) ),  #RandomRotation scale=(0.9, 1.0)
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print ('classes:', train_dataset.classes)
    # Get number of labels
    labels_length = len(train_dataset.classes)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size,image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        CustomImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, criterion2, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return
    times = []
    epoch_list=[]
    losses_list=[]
    test_losses_list=[]
    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        losses=train(train_loader, model, criterion, criterion2, optimizer, epoch, args)

        # evaluate on validation set
        acc1,loss_2,test_losses = validate(val_loader, model, criterion, criterion2, args)
        epoch_list.append(epoch+1)
        losses_list.append(losses)
        test_losses_list.append(test_losses)
        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)


        # remember best loss and save checkpoint
        is_loss = loss_2 <= best_loss
        best_loss = min(loss_2, best_loss)


        if args.arch.find('alexnet') != -1:
            pre_name = './alexnet'
        elif args.arch.find('inception_v3') != -1:
            pre_name = './inception_v3'
        elif args.arch.find('xception') != -1:
            pre_name = './xception'
        elif args.arch.find('inceptionresnetv2') != -1:
            pre_name = './inceptionresnetv2'
        elif args.arch.find('RepVGG_A1') != -1:
            pre_name = './RepVGG_A1'
        elif args.arch.find('RepVGG_B2g4') != -1:
            pre_name = './RepVGG_B2g4'
        elif args.arch.find('Transform_large') != -1:
            pre_name = './Transform_large'
        elif args.arch.find('Transform_base') != -1:
            pre_name = './Transform_base'
        elif args.arch.find('densenet121') != -1:
            pre_name = './mt2_densenet'
        elif args.arch.find('densenet201') != -1:
            pre_name = './densenet201'
        elif args.arch.find('resnet50') != -1:
            pre_name = './resnet50'
        else:
            print('### please check the args.arch for pre_name###')
            exit(-1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best,is_loss,pre_name)

        # time2 = time.time()
        # print("第",epoch,"epoch的时间为",time2-time1)
        # times.append(time2-time1)
    # PATH = pre_name + '_fundus_net.pth'
    # torch.save(model.state_dict(), PATH)
    print('Finished Training best_acc: ', best_acc1)

    with open(args.arch+"_mtl_loss.pkl","wb") as f:
        pickle.dump(losses_list,f)
        pickle.dump(test_losses_list,f)
        pickle.dump(epoch_list,f)




def train(train_loader, model, criterion, criterion2, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1_1', ':6.2f')
    top5 = AverageMeter('Acc@5_1', ':6.2f')
    top1_2 = AverageMeter('Acc@1_2', ':6.2f')
    top5_2 = AverageMeter('Acc@5_2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses1, losses2, losses, top1,
                             top5, top1_2, top5_2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
	    
		# Adding comments by Jiang Jiewei on January 9, 2024 
		# Obtaining the labels and predicted probabilities for the two-layer classifier to calculate the combined loss of their integration.
        target1=target[1] #两个类别
        target2=target[0] #五个类别
        target1 = target1.cuda(args.gpu, non_blocking=True)
        target2 = target2.cuda(args.gpu, non_blocking=True)

        output1,output2 = model(images)
        loss1 = criterion(output1, target1)
        loss2=criterion2(output2,target2)
        loss=0.2*loss1+0.8*loss2

        acc1, acc5 = accuracy(output1, target1, topk=(1, 2))
        acc1_2, acc5_2 = accuracy(output2, target2, topk=(1, 2))
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        top1_2.update(acc1_2[0], images.size(0))
        top5_2.update(acc5_2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return losses


def validate(val_loader, model, criterion, criterion2, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1_1', ':6.2f')
    top5 = AverageMeter('Acc@5_1', ':6.2f')
    top1_2 = AverageMeter('Acc@1_2', ':6.2f')
    top5_2 = AverageMeter('Acc@5_2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses1, losses2, losses, top1, top5, top1_2, top5_2,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target1=target[1]
            target2=target[0]
            target1 = target1.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

            # compute output
            output1,output2 = model(images)
            loss1 = criterion(output1, target1)
            loss2 = criterion2(output2, target2)
            loss=0.2*loss1+0.8*loss2
            # torch.cuda.synchronize()


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output1, target1, topk=(1, 2))
            acc1_2, acc5_2 = accuracy(output2, target2, topk=(1, 2))
            losses1.update(loss1.item(), images.size(0))
            losses2.update(loss2.item(), images.size(0))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top1_2.update(acc1_2[0], images.size(0))
            top5_2.update(acc5_2[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1_1 {top1.avg:.3f} Acc@5_1 {top5.avg:.3f} Acc@1_2 {top1_2.avg:.3f} Acc@5_2 {top5_2.avg:.3f}'
              .format(top1=top1, top5=top5, top1_2=top1_2, top5_2=top5_2))

    return top1_2.avg, losses2.avg,losses #保存level2最好的模型

def save_checkpoint(state, is_best, is_loss, pre_filename='my_checkpoint.pth.tar'):
    check_filename = pre_filename + 'accloss_checkpoint.pth.tar'
    des_best_filename = pre_filename + '_model_best_acc.pth.tar'
    des_best_loss_filename = pre_filename + '_model_best_loss.pth.tar'
    torch.save(state, check_filename)
    if is_best:
        shutil.copyfile(check_filename, des_best_filename)
    if is_loss:
        shutil.copyfile(check_filename, des_best_loss_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 20))
    lr_decay = 0.1 ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay
        # param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


