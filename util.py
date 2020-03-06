import torch
import shutil
import os
import numpy as np
import cv2
from main import main
from data.Scenario_data_loader import Scenario1, Scenario2, Scenario3

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    save the ckpt during model training
    :param state: model parameter
    :param is_best: Is this the current best?
    :param filename: The name for ckpt to save
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reset_folder(target="./data"):
    """remove those folders and files in case they override correct result"""
    if os.path.exists(os.path.join(target, 'train', 'invasive')):
        shutil.rmtree(os.path.join(target, 'train', 'invasive'))
        shutil.rmtree(os.path.join(target, 'train', 'noninvasive'))

    if os.path.exists(os.path.join(target, "train.txt")):
        os.remove(os.path.join(target, "train.txt"))
        os.remove(os.path.join(target, "valid.txt"))

    if os.path.exists(os.path.join("checkpoint.pth.tar")):
        os.remove(os.path.join("checkpoint.pth.tar"))

    if os.path.exists(os.path.join("model_best.pth.tar")):
        os.remove(os.path.join("model_best.pth.tar"))

    if os.path.exists(os.path.join(target, 'val', 'invasive')):
        shutil.rmtree(os.path.join(target, 'val', 'invasive'))
        shutil.rmtree(os.path.join(target, 'val', 'noninvasive'))


def compute_mean(target, image_size, affix="jpg"):
    """
    by the time to compute mean,
    we should have setup two folders, train and val,
    where two subfolders, pos/neg class should available
    if size mismatch, then pad 0 to image
    :return:
    """
    if not os.path.exists(target, "train") or not os.path.exists(target, "val"):
        print("The data folder is not available!\nProgram will quit!")
        return
    file_path = [os.path.join(r,file) for r,d,f in os.walk(r"./data/train") for file in f if file[-len(affix):]==affix]
    initial_array = np.zeros(image_size)
    for file in file_path:
        # in pytorch, transformer automatically rescale image to 0-1.
        # We do same thing at here.
        new_img = cv2.imread(file)/255.0
        new_img_cp = np.zeros(image_size)
        row_min, col_min = min(new_img.size[0], image_size[0]), min(new_img.size[1], image_size[1])
        new_img_cp += new_img[:row_min,:col_min,:]
        initial_array += new_img_cp
    return np.mean(initial_array, axis=tuple(range(initial_array.ndim-1))).tolist(), \
           np.std(initial_array, axis=tuple(range(initial_array.ndim-1))).tolist()


def EDA_image_resolution(target, pos, neg, affix="jpg"):
    """
    EDA current dataset to get some basic knowledge
    :param target: root folder
    :param pos: name of positive class
    :param neg: name of negative class
    :param affix: picture file to scan. jpg by default
    :return:
    """
    reso_dic = {}
    pos_file_path = [os.path.join(r,file) for r,d,f in os.walk(os.path.join(target, pos))
                 for file in f if file[-len(affix):]==affix]
    neg_file_path = [os.path.join(r,file) for r,d,f in os.walk(os.path.join(target, neg))
                 for file in f if file[-len(affix):]==affix]
    whole_file_path = pos_file_path+neg_file_path
    for file in whole_file_path:
        pic = cv2.imread(file)
        reso_rep = "_".join(map(str, pic.shape))
        reso_dic[reso_rep] = reso_dic.get(reso_rep,0) + 1
    print(sorted(reso_dic.items(), key=lambda x:x[-1],reverse=True))


def run_scenario1_all_year_ratio(performance_list):
    """A wrapper to train scenario 1"""
    target = "data"
    total_year = 7
    # Prepare file to train
    for year in [2012+i for i in range(1,total_year)]:
        for split_ratio in [0.3+float(i)/10 for i in range(0,5)]:
            reset_folder(target)
            print("Current ratio: "+str(split_ratio))
            print("Current year: "+str(year))
            current_scenario = Scenario1(split_ratio = split_ratio)
            current_scenario.load_multiple_year(year_list=[year])
            # call main function to train
            main()
    # print out performance , acc1 and acc5
    for idx, ele in enumerate(performance_list):
        if idx==0:
            print("copy paste average accuracy: ")
        print(round(float(ele[0].cpu()),2),round(float(ele[1].cpu()),2))
        if (idx+1)%5==0:
            print()


def tune_scenario1(args, performance_dict):
    """
    Fine-tune scenario 1
    :param args: User defined input
    :param performance_dict: record performance
    :return:
    """
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
    """
    A wrapper to train scenario 1 in single shot
    :return: performance
    """
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


def run_scenario2(performance_list, region_count = [5, 10, 15, 19]):
    target = "data"
    # only for scenario 2
    if os.path.exists(os.path.join(target, "All_invasive")):
        shutil.rmtree(os.path.join(target, "All_invasive"))
        shutil.rmtree(os.path.join(target, 'All_Noninvasive'))
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


def tune_scenario2(args, performance_dict):
    """A wrapper to train scenario 2"""
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



def tune_scenario3(args, performance_dict):
    """A wrapper to fine tune parameters involved in scenario3"""
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

def run_scenario3(train_year= [2012, 2014, 2016, 2018], valid_year= [2013, 2015, 2017]):
    """A wrapper to run scenario 3"""
    target = "data"
    print("Training year is: "+str(train_year))
    print("Validation year is: "+str(valid_year))
    current_scenario = Scenario3()
    reset_folder(target)
    current_scenario.copy_dataset(train_year, valid_year)
    main()