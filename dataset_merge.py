import glob
import cv2
import os

'''
merging WIKI dataset and meagaAge asian dataset

'''


class ImagePreprocessor:

    def __init__(self):
        self.megaage_dir = './raw_data/megaage_asian/'
        self.wiki_dir = './raw_data/wiki_crop/'
        self.completed_dir = './preprocessed_data/'

    def preprocess(self):
        self.preprocess_megaage()
        self.preprocess_wiki()

    def preprocess_megaage(self):
        mega_fname = [self.megaage_dir + 'train/' + str(name) + '.jpg' for name in range(1, 40001)]

        with open(self.megaage_dir + 'list/train_age.txt', 'r') as f:
            mega_label = f.read().split('\n')

        for i, fname in enumerate(mega_fname):

            print('preprocessing MegaAge data...' + fname)
            tmp_img = cv2.imread(fname)

            dir_name = mega_label[i]
            if not os.path.exists(self.completed_dir + 'train/' + dir_name):
                os.makedirs(self.completed_dir + 'train/' + dir_name)

            cv2.imwrite(self.completed_dir + 'train/' + mega_label[i] + '/m' + str(i) + '.jpg', tmp_img)
            if i == 8000:
                break

        mega_fname_test = [self.megaage_dir + 'test/' + str(name) + '.jpg' for name in range(1, 3946)]

        with open(self.megaage_dir + 'list/test_age.txt', 'r') as f:
            mega_label_test = f.read().split('\n')

        for i, fname in enumerate(mega_fname_test):

            print('preprocessing MegaAge data...' + fname)
            tmp_img = cv2.imread(fname)

            dir_name = mega_label_test[i]
            if not os.path.exists(self.completed_dir + 'test/' + dir_name):
                os.makedirs(self.completed_dir + 'test/' + dir_name)

            cv2.imwrite(self.completed_dir + 'test/' + mega_label_test[i] + '/m' + str(i) + '.jpg', tmp_img)
            if i == 1000:
                break

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
            if age < 0 or age > 100:
                continue

            dir_name = str(age)
            if not os.path.exists(self.completed_dir + 'train/' + dir_name):
                os.makedirs(self.completed_dir + 'train/' + dir_name)

            tmp_img = cv2.imread(fname)

            cv2.imwrite(self.completed_dir + 'train/' + dir_name + '/w' + str(i) + '.jpg', tmp_img)
            if i == 2000:
                break


def main():
    if not os.path.isfile('./out/prepro_flag.bin'):
        prepro = ImagePreprocessor()
        prepro.preprocess()
        with open('./out/prepro_flag.bin', 'wb') as f:
            pass


if __name__ == '__main__':
    main()
