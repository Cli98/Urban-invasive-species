import os, sys
from PIL import Image
import random
import shutil

random.seed(2020)
def convert2jpeg(sour_dir, dest_dir, show=False):
    """
    Convert tif image to jpeg format
    :param sour_dir: dir of source data
    :param dest_dir: dir of target data
    :param show: plot image if required
    :return: None
    """
    file_list_tif = {}
    file_list_jpeg = {}
    for sub_dir in os.listdir(sour_dir):
        file_list_tif[sub_dir] = []
        file_list_jpeg[sub_dir] = []
        if not os.path.exists(os.path.join(dest_dir,sub_dir)):
            os.mkdir(os.path.join(dest_dir,sub_dir))
        for file in os.listdir(os.path.join(sour_dir, sub_dir)):
            if file[-3:] == "tif":
                # print(file)
                outfile = file[:-3] + "jpg"
                file_list_tif[sub_dir].append(os.path.join(sour_dir,sub_dir,file))
                file_list_jpeg[sub_dir].append(os.path.join(dest_dir,sub_dir,outfile))
                try:
                    im = Image.open(os.path.join(sour_dir,sub_dir,file))
                except:
                    print(os.path.join(sour_dir,sub_dir,file),' cannot open!')
                else:
                    out = im.convert("RGB")
                    out.save(os.path.join(dest_dir,sub_dir,outfile), "JPEG", quality=100)
                if show:
                    out.show()

def split_dataset(positive, negative, target, year=None, affix = None, rate = 0.6):
    """
    Given a dataset of jpeg image, prepare train and validation
    dataset for hold out validation
    :param positive: path to save positive samples
    :param negative: path to save negative samples
    :param target: parent directory to save data
    :return:
    """
    # possible input 1: positive --> a list of filepaths available
    # possible input 2: positive --> dir of data folder
    if isinstance(positive, list):
        positive_list = positive
    if isinstance(negative, list):
        negative_list = negative
    # only need pos/neg spot for overlap checking purpose
    # pos/neg spot saves name of file(exclude affix)
    positive_spot = []
    negative_spot = []

    # DataLoader will pass year to this function, so year by default is not None
    # If you decide to use your own pipeline, then consider to name your folder
    # with the format of "year_class_affix". Affix example: "*.jpeg"
    if not year:
        # Assume current folder's name in format year_class_affix
        thisYear = str(positive).split('_')[0].split('/')[1]
    else:
        thisYear = year
    #print("pos:",positive)
    if isinstance(positive, str):
        for file in os.listdir(positive):
            #if file[-4:] == '.jpg' and '_' not in file:
            if file[-4:] == affix:
                positive_list.append(os.path.join(positive,file))
                positive_spot.append(file.split('.')[0])
        print('positive:',len(positive_spot))

    if isinstance(negative, str):
        for file in os.listdir(negative):
            #if file[-4:] == '.jpg' and '_' not in file:
            if file[-4:] == affix:
                negative_list.append(os.path.join(negative,file))
                negative_spot.append(file.split('.')[0])
        print('negative:',len(negative_spot))

        print('overlapping:', check_overlapping(positive_spot, negative_spot))

    train_list, valid_list = split_train_valid(positive_list, negative_list, rate=rate)
    # print to file for generated train/val data
    with open(os.path.join(target, 'train.txt'), 'w') as fp:
        for sample in train_list:
            fp.write(str(sample[0])+'\t'+str(sample[1])+'\n')
    with open(os.path.join(target, 'valid.txt'), 'w') as fp:
        for sample in valid_list:
            fp.write(str(sample[0])+'\t'+str(sample[1])+'\n')
    # Generate dataset for train/val dataset, with given class(invasive or non-invasive)
    # with the help of data loader, only need to add each year of data in iterative manner
    if not os.path.exists(os.path.join(target, 'train', 'invasive')):
        os.makedirs(os.path.join(target, 'train', 'invasive'),exist_ok=False)
        os.makedirs(os.path.join(target, 'train', 'noninvasive'), exist_ok=False)
    if not os.path.exists(os.path.join(target, 'val', 'invasive')):
        os.makedirs(os.path.join(target, 'val', 'invasive'), exist_ok=False)
        os.makedirs(os.path.join(target, 'val', 'noninvasive'), exist_ok=False)

    for sample in train_list:
        newName = str(thisYear) + '_' + sample[1].split(r"/")[-1]
        # print(newName)
        if sample[0] == 1:
            shutil.copy(sample[1], os.path.join(target, 'train', 'invasive', newName))
        else:
            shutil.copy(sample[1], os.path.join(target, 'train', 'noninvasive', newName))

    for sample in valid_list:
        newName = str(thisYear) + '_' + sample[1].split(r"/")[-1]
        if sample[0] == 1:
            shutil.copy(sample[1], os.path.join(target, 'val', 'invasive', newName))
        else:
            shutil.copy(sample[1], os.path.join(target, 'val', 'noninvasive', newName))

def split_train_valid(positive_list, negative_list, rate = 0.6):
    """
    Modified train-test split method for the propose of current project
    :param positive_list: filepath for positive samples
    :param negative_list: filepath for negative samples
    :param rate: ratio of samples in training set
    :return: A list of [class, filepath] for train/val dataset
    """
    train_list = []
    valid_list = []
    random.shuffle(positive_list)
    random.shuffle(negative_list)
    for i, positive in enumerate(positive_list):
        if i < rate*len(positive_list):
            # here 1 for positive sample
            train_list.append([1,positive])
        else:
            valid_list.append([1,positive])
    for i, negative in enumerate(negative_list):
        if i < rate*len(negative_list):
            # here 0 for negative sample
            train_list.append([0,negative])
        else:
            valid_list.append([0,negative])
    return train_list, valid_list


def check_overlapping(positive_list, negative_list):
    """
    Double check the result of dataset to see for any possible overlap
    Consider to use set for efficiency
    :param positive_list: filepath for positive samples
    :param negative_list: filepath for negative samples
    :return: overlap count for two list
    """
    count = 0
    for spot in positive_list:
        if spot in negative_list:
            count += 1
    return count

if __name__=='__main__':
    root = '.'
    #convert2jpeg(sour_dir='/data/jeff-Dataset/invasive/tif_dataset', dest_dir='/data/jeff-Dataset/invasive/jpg_dataset/',show=False)
    split_dataset(os.path.join(root, 'Invasive_jpeg'), os.path.join(root, 'Noninvasive_jpeg'), root)
