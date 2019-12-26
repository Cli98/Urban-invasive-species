import torch
import shutil
import os


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

