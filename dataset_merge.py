import glob
import os
import shutil
from random import shuffle

'''
merging WIKI dataset and meagaAge asian dataset

'''


class ImagePreprocessor:

    def __init__(self):
        self.megaageasian_dir = './raw_data/megaage_asian/'
        self.megaage_dir = './raw_data/megaage/'
        self.wiki_dir = './raw_data/wiki_crop/'
        self.utk_dir = './raw_data/UTKFace.tar/UTKFace/'
        self.completed_dir = './preprocessed_data/'

    def preprocess(self):
        self.preprocess_UTK()
        # self.preprocess_megaage_asian()
        # self.preprocess_megaage()
        # self.preprocess_wiki()

    def preprocess_megaage(self):
        mega_fname = [self.megaage_dir + 'train/' + str(name) + '.jpg' for name in range(8531, 41942)]

        with open(self.megaage_dir + 'list/train_age.txt', 'r') as f:
            mega_label = f.read().split('\n')

        for i, fname in enumerate(mega_fname):
            break
            print('preprocessing MegaAge data...' + fname)

            dir_name = mega_label[i]
            if not os.path.exists(self.completed_dir + 'train/' + dir_name):
                os.makedirs(self.completed_dir + 'train/' + dir_name)

            dst = self.completed_dir + 'train/' + mega_label[i] + '/m' + str(i) + '.jpg'
            shutil.copy2(fname, dst)

        mega_fname_test = [self.megaage_dir + 'test/' + str(name) + '.jpg' for name in range(1, 8530)]

        with open(self.megaage_dir + 'list/test_age.txt', 'r') as f:
            mega_label_test = f.read().split('\n')

        for i, fname in enumerate(mega_fname_test):

            print('preprocessing MegaAge data...' + fname)

            if i < 4000:
                dir_name = mega_label_test[i]
                if not os.path.exists(self.completed_dir + 'test/' + dir_name):
                    os.makedirs(self.completed_dir + 'test/' + dir_name)
                dst = self.completed_dir + 'test/' + mega_label_test[i] + '/m' + str(i) + '.jpg'

                shutil.copy2(fname, dst)
            else:
                dir_name = mega_label_test[i]
                if not os.path.exists(self.completed_dir + 'validation/' + dir_name):
                    os.makedirs(self.completed_dir + 'validation/' + dir_name)
                dst = self.completed_dir + 'validation/' + mega_label_test[i] + '/m' + str(i) + '.jpg'

                shutil.copy2(fname, dst)

    def preprocess_megaage_asian(self):
        mega_fname = [self.megaageasian_dir + 'train/' + str(name) + '.jpg' for name in range(1, 40001)]

        with open(self.megaageasian_dir + 'list/train_age.txt', 'r') as f:
            mega_label = f.read().split('\n')

        for i, fname in enumerate(mega_fname):

            print('preprocessing MegaAge data...' + fname)

            if i < 36000:
                dir_name = mega_label[i]
                if not os.path.exists(self.completed_dir + 'train/' + dir_name):
                    os.makedirs(self.completed_dir + 'train/' + dir_name)

                dst = self.completed_dir + 'train/' + mega_label[i] + '/m' + str(i) + '.jpg'
                shutil.copy2(fname, dst)
            else:
                dir_name = mega_label[i]
                if not os.path.exists(self.completed_dir + 'validate/' + dir_name):
                    os.makedirs(self.completed_dir + 'validate/' + dir_name)

                dst = self.completed_dir + 'validate/' + mega_label[i] + '/m' + str(i) + '.jpg'
                shutil.copy2(fname, dst)

        mega_fname_test = [self.megaageasian_dir + 'test/' + str(name) + '.jpg' for name in range(1, 3946)]

        with open(self.megaageasian_dir + 'list/test_age.txt', 'r') as f:
            mega_label_test = f.read().split('\n')

        for i, fname in enumerate(mega_fname_test):

            print('preprocessing MegaAge data...' + fname)

            dir_name = mega_label_test[i]
            if not os.path.exists(self.completed_dir + 'test/' + dir_name):
                os.makedirs(self.completed_dir + 'test/' + dir_name)
            dst = self.completed_dir + 'test/' + mega_label_test[i] + '/ma' + str(i) + '.jpg'

            shutil.copy2(fname, dst)

    def preprocess_wiki(self):

        wiki_fname = glob.iglob(self.wiki_dir + '**/*.jpg', recursive=True)

        for i, fname in enumerate(wiki_fname):
            print('preprocessing Wiki data...' + fname)

            date_info = fname.split('_')[3:]
            print(date_info)

            try:
                birth_year = int(date_info[0][:4])
                pictured_date = int(date_info[1][:-4])
            except ValueError:
                continue

            age = pictured_date - birth_year
            if age <= 0 or age > 100:
                continue

            dir_name = str(age)
            if not os.path.exists(self.completed_dir + 'train/' + dir_name):
                os.makedirs(self.completed_dir + 'train/' + dir_name)

            shutil.copy2(fname, self.completed_dir + 'train/' + dir_name + '/w' + str(i) + '.jpg')

    def preprocess_UTK(self):

        utk_fname = glob.iglob(self.utk_dir + '**/*.jpg', recursive=True)
        utk_fname = list(utk_fname)

        for i in range(3):
            shuffle(utk_fname)

        for i, fname in enumerate(utk_fname):
            print('preprocessing utk data...' + fname)
            age_str = fname.split('_')[1][25:]
            print(age_str)
            age = int(age_str)
            if age > 100:
                age = 100
            age_str=str(age)

            if i < 20000:
                dir = self.completed_dir + 'train/'
            elif i < 22000:
                dir = self.completed_dir + 'validate/'
            else:
                dir = self.completed_dir + 'test/'

            if not os.path.exists(dir + age_str):
                os.makedirs(dir + age_str)

            shutil.copy2(fname, dir + age_str + '/u' + str(i) + '.jpg')


def main():
    if not os.path.isfile('./out/prepro_flag.bin'):
        prepro = ImagePreprocessor()
        prepro.preprocess()


if __name__ == '__main__':
    main()
