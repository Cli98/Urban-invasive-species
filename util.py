import torch
import shutil
import os
import numpy as np
import cv2


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
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


