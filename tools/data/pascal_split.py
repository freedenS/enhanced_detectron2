import os
import random
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='This script support spliting voc data to train, val and test set.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to annotation files directory.')
    parser.add_argument('--trainval_ratio', type=float, default=None,
                        help='ratio of train and val set.')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='ratio of train set.')
    args = parser.parse_args()

    total_xml = os.listdir(os.path.join(args.data_dir, 'Annotations'))

    num=len(total_xml)
    list=range(num)
    tv=int(num*args.trainval_ratio)
    tr=int(tv*args.train_ratio)
    trainval= random.sample(list,tv)
    train=random.sample(trainval,tr)

    ftrainval = open(os.path.join(args.data_dir, 'ImageSets/Main/trainval.txt'), 'w')
    ftest = open(os.path.join(args.data_dir, 'ImageSets/Main/test.txt'), 'w')
    ftrain = open(os.path.join(args.data_dir, 'ImageSets/Main/train.txt'), 'w')
    fval = open(os.path.join(args.data_dir, 'ImageSets/Main/val.txt'), 'w')

    for i  in list:
        name=total_xml[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()


if __name__ == '__main__':
    main()