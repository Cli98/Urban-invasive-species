import argparse
import os
import random
import time
import warnings
import shutil
import sys
sys.path.append("./data")

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
from meter import AverageMeter, ProgressMeter
from util import save_checkpoint, adjust_learning_rate, accuracy, reset_folder, compute_mean
from Scenario_data_loader import Scenario1, Scenario2, Scenario3

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#TODO: 1. modify input pipeline, add case 2
#TODO: 2. modify main, split functionality
#TODO: 3. create 3 case loader, split out preprocessing scripy
#TODO: 4. create file for meters, Done
#TODO: 5. Rest to finish
#parameters setting:
parser = argparse.ArgumentParser(description='PyTorch Invasion project')
parser.add_argument('data', metavar='DIR', default="./data", type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1e7, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=2020, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

Debug = True
if Debug:
    # debug mode ,nothing to do with main code
    best_acc1 = 0
    performance_list = []
    performance_dict = dict()
args = parser.parse_args()

# Discuss those parameters in paper: seed
def main():
    """
    Setup all parameters to run deep learning model.
    Train the deep learning model as discussed in paper. Here three scenarios are presented.
    :return:
    """
    if args.seed is not None:
        # seed the RNG for all devices (both CPU and CUDA)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        # args.gpu is none, then use all gpus if possible
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        # setup word_size for distributed training
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
    """
    Worker function to train model
    :param gpu: Number of GPU to use, default 0, if NONE, then model uses all gpus
    :param ngpus_per_node: available number of gpu to use per machine. This only works in distributed setup
    :param args: parameter group. Required to train your model
    :return: None
    """

    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # single machine with multiple GPU does not falls into this branch
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if "model" in locals():
        print("Remove available model")
        del model
    if "model" in globals():
        print("Remove available model")
        del model

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        #print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

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
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
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
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # Use all gpus if args.gpu is None
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # TODO: create a function to reload checkpoint
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

    # set to true to enable internal code optimization
    cudnn.benchmark = True

    # Data loading code
    # TODO: modify code based on 3 cases
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    img_size = (200, 200)
    train_mean, train_std = compute_mean(args.data, img_size)
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=train_mean,
                                     std=train_std)
    # TODO: discuss argumentation applied at here
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # learning rate decay by 10 after each 30 epochs
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if epoch == args.epochs-1:
            print(' Validation finished! Avg stats: Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
                  .format(top1=acc1, top5=acc5))
            performance_list.append([acc1, acc5])
            performance_dict[str(args.momentum) + " " + str(args.batch_size) + " " + \
                             str(args.epochs) + " " + str(args.lr)] = round(float(acc1.cpu()),2)
            print(sorted(performance_dict.items(), key=lambda x: x[-1], reverse=True)[0])

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        #print(' Validation finished! Avg stats: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def run_scenario1_all_year_ratio():
    target = "data"
    total_year = 7
    for year in [2012+i for i in range(1,total_year)]:
        for split_ratio in [0.3+float(i)/10 for i in range(0,5)]:
            reset_folder(target)
            print("Current ratio: "+str(split_ratio))
            print("Current year: "+str(year))
            current_scenario = Scenario1(split_ratio = split_ratio)
            current_scenario.load_multiple_year(year_list=[year])
            main()

    for idx, ele in enumerate(performance_list):
        if idx==0:
            print("copy paste average accuracy: ")
        print(round(float(ele[0].cpu()),2),round(float(ele[1].cpu()),2))
        if (idx+1)%5==0:
            print()


def tune_scenario1():
    para_tune = {"batch_sizes":[32, 64, 128, 256],
                 "learning_rates":[1e-2, 1e-3, 1e-4, 1e-5],
                 "momentums":[0.9, 0.95, 0.99],
                "epochs":[30,60,90]}

    progress = 1
    total_comb = 4*4*3*3
    for momentum in para_tune["momentums"]:
        for lr in para_tune["learning_rates"]:
            for epoch in para_tune["epochs"]:
                for batch_size in para_tune["batch_sizes"]:
                    print("Current parameter setting: momentum batch_size epochs lr")
                    print(momentum, batch_size, epoch, lr)
                    args.momentum = momentum
                    args.batch_size = batch_size
                    args.epochs = epoch
                    args.lr = lr
                    scenario1_wrapper()
                    print("Current progress: ", str(progress) , "/" , str(total_comb))
                    progress+=1

    print(sorted(performance_dict.items(),key= lambda x:x[-1],reverse=True))


def scenario1_wrapper():
    # fix split ratio as 0.7
    # fix year for all year
    target = "data"
    total_year = 7
    year_list = [2012+i for i in range(total_year)]
    split_ratio = 0.7
    reset_folder(target)
    current_scenario = Scenario1(split_ratio = split_ratio)
    current_scenario.load_multiple_year(year_list)
    main()


def run_scenario2():
    target = "data"
    # only for scenario 2
    if os.path.exists(os.path.join(target, "All_invasive")):
        shutil.rmtree(os.path.join(target, "All_invasive"))
        shutil.rmtree(os.path.join(target, 'All_Noninvasive'))
    region_count = [5, 10, 15, 19]
    # following three lines target All_invasive/All_noninvasive folder
    data_loader = Scenario2()
    data_loader.prepare_dataset()
    data_loader.get_available_region()
    for count in region_count:
        reset_folder(target)
        train_region, val_region = data_loader.get_training_region(k=count)
        data_loader.complete_dataset(train_region, val_region)
        main()

    for idx, ele in enumerate(performance_list):
        if idx==0:
            print("copy paste average accuracy: ")
        print(round(float(ele[0].cpu()),2),round(float(ele[1].cpu()),2))


def tune_scenario2():
    target = "data"
    para_tune = {"batch_sizes":[32, 64, 128, 256],
                 "learning_rates":[1e-2, 1e-3, 1e-4, 1e-5],
                 "momentums":[0.9, 0.95, 0.99],
                "epochs":[30,60,90]}
    # only for scenario 2
    if os.path.exists(os.path.join(target, "All_invasive")):
        shutil.rmtree(os.path.join(target, "All_invasive"))
        shutil.rmtree(os.path.join(target, 'All_Noninvasive'))
    region_count = 10
    # following three lines target All_invasive/All_noninvasive folder
    data_loader = Scenario2()
    data_loader.prepare_dataset()
    data_loader.get_available_region()
    progress = 1
    total_comb = 4*4*3*3
    for momentum in para_tune["momentums"]:
        for lr in para_tune["learning_rates"]:
            for epoch in para_tune["epochs"]:
                for batch_size in para_tune["batch_sizes"]:
                    print("Current parameter setting: momentum batch_size epochs lr")
                    print(momentum, batch_size, epoch, lr)
                    args.momentum = momentum
                    args.batch_size = batch_size
                    args.epochs = epoch
                    args.lr = lr
                    reset_folder(target)
                    train_region, val_region = data_loader.get_training_region(k=region_count)
                    data_loader.complete_dataset(train_region, val_region)
                    main()
                    print("Current progress: ", str(progress) , "/" , str(total_comb))
                    progress+=1

    print(sorted(performance_dict.items(),key= lambda x:x[-1],reverse=True))



def tune_scenario3():
    para_tune = {"batch_sizes":[32, 64, 128, 256],
                 "learning_rates":[1e-2, 1e-3, 1e-4, 1e-5],
                 "momentums":[0.9, 0.95, 0.99],
                "epochs":[30,60,90]}

    progress = 1
    total_comb = 4*4*3*3
    for momentum in para_tune["momentums"]:
        for lr in para_tune["learning_rates"]:
            for epoch in para_tune["epochs"]:
                for batch_size in para_tune["batch_sizes"]:
                    print("Current parameter setting: momentum batch_size epochs lr")
                    print(momentum, batch_size, epoch, lr)
                    args.momentum = momentum
                    args.batch_size = batch_size
                    args.epochs = epoch
                    args.lr = lr
                    run_scenario3()
                    print("Current progress: ", str(progress), "/", str(total_comb))
                    progress += 1

    print(sorted(performance_dict.items(), key=lambda x: x[-1], reverse=True))

def run_scenario3():
    target = "data"
    train_year, valid_year = [2012, 2014, 2016, 2018], [2013, 2015, 2017]
    #train_year, valid_year = [2012, 2013, 2014, 2015], [2016, 2017, 2018]
    current_scenario = Scenario3()
    current_scenario.copy_dataset(train_year, valid_year)
    reset_folder(target)
    main()


if __name__ == '__main__':
    # for S1 and S3
    # args.momentum = 0.9
    # args.batch_size = 32
    # args.epochs = 90
    # args.lr = 0.01
    # target = "data"
    # total_year = 7
    # year_list = [2012+i for i in range(7)]
    tune_scenario3()




