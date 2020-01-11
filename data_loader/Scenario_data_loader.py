import os
import random
import shutil
import sys
sys.path.append("..")
from preproc import split_dataset
class DataLoader:
    def __init__(self, root_path=".", split_ratio = 0.6):
        self.root = root_path
        self.pos_label = "invasive"
        self.neg_label = "Noninvasive"
        self.image_format = "jpg"
        self.split_ratio = split_ratio
        if not os.path.exists(self.root):
            print("Cannot find data folder in given location!\nInitialization failed!\n")
            return


    def find_data_by_year(self, year, pos, affix=None):
        if not affix and self.image_format:
            affix = self.image_format
        file_loc = []
        if pos:
            folder = os.path.join(self.pos_label, str(year)+"_"+self.pos_label+"_"+affix)
        else:
            folder = os.path.join(self.neg_label, str(year) + "_" + self.neg_label + "_" + affix)
        if not os.path.exists(folder):
            print("Cannot locate data for current year: "+str(year))
            print("The folder path: "+folder)
        else:
            #print(os.listdir(folder)[0],os.listdir(folder)[0][-len(affix):])
            file_loc = [os.path.join(folder, path) for path in os.listdir(folder) if
                        path and path[-len(affix):].lower()==affix]
        return file_loc if len(file_loc)>0 else None


    def find_data_by_region(self, region_list, pos, affix=None):
        if not affix and self.image_format:
            affix = self.image_format
        file_loc = []
        if pos:
            folder = os.path.join("All_"+self.pos_label)
        else:
            folder = os.path.join("All_"+self.neg_label)
        if not os.path.exists(folder):
            print("Cannot locate data for current scenario")
            print("The folder path you provided: " + folder)
        else:
            file_loc = [os.path.join(folder, path) for path in os.listdir(folder) if
                        path and path[-len(affix):].lower() == affix and
                        ''.join(x for x in path.split(r"/")[-1].split("_")[1].lower() if x.isalpha()) in
                        region_list]
        return file_loc if len(file_loc) > 0 else None




class Scenario1(DataLoader):
    """
    For multiple year, fill year-list with all year required
    else, fill year list with single year
    Example to use this loader:
    #year_list = [2012+i for i in range(7)]
    #current_scenario = Scenario1()
    #current_scenario.load_multiple_year(year_list)
    """
    def load_single_year(self,year=2012):
        pos_file_loc = self.find_data_by_year(year,1)
        neg_file_loc = self.find_data_by_year(year,0)
        split_dataset(pos_file_loc, neg_file_loc, self.root, year, affix="jpg", rate = self.split_ratio)
        #Expected to see 4 folders. year=2012, train/val, train/val filename generated in same folder
        #The total amount = pos/neg folder amount, pass test


    def load_multiple_year(self,year_list):
        """
        create data set for given year.
        :param year_list: a list of year to indicate which year to load
        :return:
        """
        if not isinstance(year_list, list):
            print("Year is expected to be a list\nProgram will quit\n")
            return
        for year in year_list:
            self.load_single_year(year)


class Scenario2(DataLoader):
    """

    """
    def __init__(self, root_path=".", split_ratio = 0.6, seed = 2020):
        super().__init__(root_path=root_path, split_ratio = split_ratio)
        self.avail_region = set()
        self.seed = seed
        return


    def load_single_year_data(self,year=2012):
        pos_file_loc = self.find_data_by_year(year,1)
        neg_file_loc = self.find_data_by_year(year,0)
        return pos_file_loc, neg_file_loc


    def prepare_dataset(self, available_year = [2012+i for i in range(7)], target=None, affix="jpg"):
        if not target:
            target = self.root
        if not os.path.exists(os.path.join(target, 'All_invasive')):
            os.makedirs(os.path.join(target, 'All_invasive'), exist_ok=False)
            os.makedirs(os.path.join(target, 'All_noninvasive'), exist_ok=False)
        for year in available_year:
            pos_file_loc, neg_file_loc = self.load_single_year(year)
            for file in pos_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'All_invasive', newName))
                self.avail_region.add(''.join(x for x in file.split(r"/")[-1].split("_")[0].lower() if x.isalpha()))
            for file in neg_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'All_noninvasive', newName))
                self.avail_region.add(''.join(x for x in file.split(r"/")[-1].split("_")[0].lower() if x.isalpha()))
        return


    def get_available_region(self):
        print("Total number of regions are : "+str(len(self.avail_region)))
        print("The name of regions are: "+str(self.avail_region))
        return


    def get_training_region(self, k=5):
        val_region = sorted(self.avail_region)
        if len(val_region)<=k:
            print("The total number of available region is less than required amount")
            print("Not able to get train region")
            return
        random.seed(self.seed)
        random.shuffle(val_region)
        train_region = [val_region.pop(0) for _ in range(k)]
        assert len(train_region) == k
        return train_region, val_region


    def complete_dataset(self, train_region, val_region, target=None, affix="jpg"):
        if not target:
            target = self.root
        if not os.path.exists(os.path.join(target, 'train', 'invasive')):
            os.makedirs(os.path.join(target, 'train', 'invasive'), exist_ok=False)
            os.makedirs(os.path.join(target, 'train', 'noninvasive'), exist_ok=False)
        if not os.path.exists(os.path.join(target, 'val', 'invasive')):
            os.makedirs(os.path.join(target, 'val', 'invasive'), exist_ok=False)
            os.makedirs(os.path.join(target, 'val', 'noninvasive'), exist_ok=False)
        train_region_pos = data_loader.find_data_by_region(train_region, 1)
        train_region_neg = data_loader.find_data_by_region(train_region, 0)
        val_region_pos = data_loader.find_data_by_region(val_region, 1)
        val_region_neg = data_loader.find_data_by_region(val_region, 0)
        for file in train_region_pos:
            newName = file.split(r"/")[-1]
            shutil.copy(file, os.path.join(target, 'train', 'invasive', newName))
        for file in train_region_neg:
            newName = file.split(r"/")[-1]
            shutil.copy(file, os.path.join(target, 'train', 'noninvasive', newName))
        for file in val_region_pos:
            newName = file.split(r"/")[-1]
            shutil.copy(file, os.path.join(target, 'val', 'invasive', newName))
        for file in val_region_neg:
            newName = file.split(r"/")[-1]
            shutil.copy(file, os.path.join(target, 'val', 'noninvasive', newName))
        return


class Scenario3(DataLoader):
    """
    Example for loader:
    Fill in year to load data, then call function
    train_year, valid_year = [2012, 2014, 2016, 2018], [2013, 2015, 2017]
    train_year, valid_year = [2012, 2013, 2014, 2015], [2016, 2017, 2018]
    current_scenario = Scenario3()
    current_scenario.copy_dataset(train_year, valid_year)
    """
    def copy_dataset(self, train_year, valid_year, target=None, affix="jpg"):
        if not target:
            target = self.root
        if not os.path.exists(os.path.join(target, 'train', 'invasive')):
            os.makedirs(os.path.join(target, 'train', 'invasive'), exist_ok=False)
            os.makedirs(os.path.join(target, 'train', 'noninvasive'), exist_ok=False)
        if not os.path.exists(os.path.join(target, 'val', 'invasive')):
            os.makedirs(os.path.join(target, 'val', 'invasive'), exist_ok=False)
            os.makedirs(os.path.join(target, 'val', 'noninvasive'), exist_ok=False)
        for year in train_year:
            pos_file_loc, neg_file_loc = self.load_single_year(year)
            for file in pos_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'train', 'invasive', newName))
            for file in neg_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'train', 'noninvasive', newName))
        for year in valid_year:
            pos_file_loc, neg_file_loc = self.load_single_year(year)
            for file in pos_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'val', 'invasive', newName))
            for file in neg_file_loc:
                newName = str(year) + '_' + file.split(r"/")[-1]
                shutil.copy(file, os.path.join(target, 'val', 'noninvasive', newName))


    def load_single_year(self,year=2012):
        pos_file_loc = self.find_data_by_year(year,1)
        neg_file_loc = self.find_data_by_year(year,0)
        return pos_file_loc, neg_file_loc

if __name__ == "__main__":
    #print(sys.path)
    data_loader = Scenario2()
    data_loader.prepare_dataset()
    data_loader.get_available_region()
    train_region, val_region = data_loader.get_training_region(k=5)
    data_loader.complete_dataset(train_region, val_region)














