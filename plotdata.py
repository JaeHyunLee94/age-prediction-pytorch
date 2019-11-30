import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_data_districution():
    train_path = './preprocessed_data/train/'
    validate_path = './preprocessed_data/validate/'
    test_path = './preprocessed_data/test/'

    y = np.array([])

    for i in range(100):
        age_str = str(i + 1)
        num = 0
        if os.path.isdir(train_path + age_str):
            num += len(os.listdir(train_path + age_str))
        if os.path.isdir(validate_path + age_str):
            num += len(os.listdir(validate_path + age_str))
        if os.path.isdir(test_path + age_str):
            num += len(os.listdir(test_path + age_str))
        y = np.concatenate((y, np.array([i] * num)))

    sns.set(color_codes=True)
    sns.distplot(y)

    plt.title('Megaage_asian+WIKI_crop')
    plt.axis([-10,110,0,0.0525])
    plt.xlabel('age')
    plt.ylabel('Num of Data')
    plt.savefig('./out/Megaage_asian_wikicrop.jpg')
    plt.show()


if __name__ == '__main__':
    plot_data_districution()
