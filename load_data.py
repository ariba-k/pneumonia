import pandas as pd
import collections
import datetime as dt
import numpy as np
from glob import glob
import os, sys
import matplotlib.pyplot as plt
from skimage.transform import resize
from lxml import objectify
from augment_data import *
from pprint import pprint
import swifter
import cv2
import pickle
import tables
import random
from enum import Enum


# removed: 'subcutaneous emphysema': 8, 'Hernia': 10, 'hiatal hernia': 10, humeral fracture: 21, vertebral fracture: 21
# TODO heart insufficiency, pseudonodule, total atelectasis, lepidic adenocarcinoma, COPD signs =? emphesyma
# TODO remove humeral fracture, vertebral fracture
# TODO check pleural other

# TODO metagrouping
LABEL_NORM_MAP = {'NA': -1, 'No Finding': 0, 'normal': 0, 'Pneumonia': 1, 'Consolidation': 2, 'Infiltration': 3,
                  'infiltrates': 3, 'Pneumothorax': 4, 'Effusion': 5, 'Pleural Effusion': 5,
                  'loculated pleural effusion': 5, 'Pleural_Thickening': 6,
                  'pleural thickening': 6, 'apical pleural thickening': 6, 'calcified pleural thickening': 6,
                  'Edema': 7, 'pulmonary edema': 7, 'Emphysema': 8, 'Fibrosis': 9, 'pulmonary fibrosis': 9,
                  'soft tissue mass': 10, 'Nodule': 11, 'pseudonodule': 11, 'Mass': 12, 'pulmonary mass': 12,
                  'mediastinal mass': 12, 'pleural mass': 12, 'Cardiomegaly': 13, 'Atelectasis': 14,
                  'segmental atelectasis': 14, 'round atelectasis': 14, 'laminar atelectasis': 14,
                  'lobar atelectasis': 14, 'total atelectasis': 14, 'Enlarged Cardiomediastinum': 15,
                  'heart insufficiency': 16, 'Lung Opacity': 17, 'Lung Lesion': 18, 'lepidic adenocarcinoma': 19,
                  'Pleural Other': 20, 'Fracture': 21, ' callus rib fracture': 21,
                  'rib fracture': 21, 'Atherosclerosis': 22, 'COPD signs': 23, 'lymphangitis carcinomatosa': 24,
                  'pulmonary artery hypertension': 25,  'pulmonary hypertension': 25, 'tuberculosis': 26,
                  'tuberculosis sequelae': 27, 'lung metastasis': 28, 'post radiotherapy changes': 29,
                  'atypical pneumonia': 30, 'bone metastasis': 31, 'respiratory distress': 32, 'asbestosis signs': 33}

LABEL_THE_14 = {'NA': -1, 'No Finding': 0, 'normal': 0, 'Pneumonia': 1, 'Consolidation': 2, 'Infiltration': 3,
                'infiltrates': 3, 'Pneumothorax': 4, 'Effusion': 5, 'Pleural Effusion': 5,
                'loculated pleural effusion': 5, 'Pleural_Thickening': 6,
                'pleural thickening': 6, 'apical pleural thickening': 6, 'calcified pleural thickening': 6,
                'Edema': 7, 'pulmonary edema': 7, 'Emphysema': 8, 'Fibrosis': 9, 'pulmonary fibrosis': 9,
                'Hernia': 10, 'hiatal hernia': 10, 'Nodule': 11,
                'pseudonodule': 11, 'Mass': 12, 'pulmonary mass': 12,
                'mediastinal mass': 12, 'pleural mass': 12, 'Cardiomegaly': 13, 'Atelectasis': 14,
                'segmental atelectasis': 14, 'round atelectasis': 14, 'laminar atelectasis': 14,
                'lobar atelectasis': 14, 'total atelectasis': 14}

LABEL_NORM_MAP = LABEL_THE_14  # FIXME

# lowercases the keys of LABEL_NORM_MAP
LABEL_NORM_MAP = {k.lower().replace(' ', ''): v for k, v in LABEL_NORM_MAP.items()}

# allows label output based on dict value
LABEL_REV_MAP = {int(lval): lkey for lkey, lval in LABEL_NORM_MAP.items()}
for lkey, lval in LABEL_NORM_MAP.items():
    LABEL_REV_MAP[lval] = lkey

# maps each label to an integer
LEFT_OUT = set()


def label_mapping(key):
    global LEFT_OUT
    keyy = key.lower().replace(' ', '')
    if keyy in LABEL_NORM_MAP:
        return LABEL_NORM_MAP[keyy]
    LEFT_OUT.add(key)
    return -1


def load_img(img_path, img_dim=(512, 512)):
    img = plt.imread(img_path)
    img = resize(img, img_dim)
    img = img[:, :, 0] if len(img.shape) == 3 else img
    return img  # normalize separately as we can't normalize before augmentation


# converts female, male to 0, 1; man if unknown (there are barely any unknowns)
def convert_sex_to_bin(sexes):
    return [int(str(s).lower()[0] == 'f') for s in sexes]


# DATASETS - loads image paths with labels
def cxr14_load_labels(path):
    # global CXR14LABELS
    labels_df = pd.read_csv(path, index_col=False)
    img_name_col = labels_df['Image Index']
    label_col = labels_df['Finding Labels']
    sex_col = convert_sex_to_bin(labels_df['Patient Gender'])
    # note: removed pixel spacing and size because unintelligible arguments
    extra = [{'age': age, 'sex': sex} for age, sex in zip(labels_df['Patient Age'], sex_col)]

    labels = {}
    for name, label, e in zip(img_name_col, label_col, extra):
        label = [label_mapping(l) for l in label.split('|')]
        labels[name] = {'label': label, 'extra': e}

    return labels


def cxr14_load_imgpath_w_labels(labels_path, img_path):
    labels = cxr14_load_labels(labels_path)
    img_paths = [img_path for img_path in glob(img_path + 'images/*.png')]
    return [[img, labels[os.path.basename(img)]['label'], labels[os.path.basename(img)]['extra']] for img in img_paths]


def chexpert_load_labels(path):  # note: use for mimic database #sex_dict #path(img_name)
    labels_train_df = pd.read_csv(path + '/train.csv', index_col=False)
    labels_valid_df = pd.read_csv(path + '/valid.csv', index_col=False)
    labels_df = pd.concat([labels_train_df, labels_valid_df])
    img_name_col = list(labels_df['Path'])
    first_col = list(labels_df.columns.values).index('No Finding')
    last_col = list(labels_df.columns.values).index('Support Devices') + 1  # included
    disease_df = labels_df.iloc[:, first_col:last_col]
    label_col = [list(disease_df.columns[np.where(i)]) for i in disease_df.values == 1]
    # TODO FIXME implement certainty values: blank: unmentioned, 0: negative, -1: not certain, 1: positive

    if 'Age' in labels_df.columns:  # chexpert
        sex_col = convert_sex_to_bin(labels_df['Sex'])
        extra = [{'age': age, 'sex': sex, 'position': pos, 'view': view}
                 for age, sex, pos, view in zip(labels_df['Age'], sex_col, labels_df['AP/PA'], labels_df['Frontal/Lateral'])]
    else:  # mimic
        extra = [{'view': view} for view in labels_df['View']]

    labels = {}
    for img_path, label, e in zip(img_name_col, label_col, extra):
        label = [label_mapping(l) for l in label]
        labels[img_path] = {'label': label, 'extra': e}
    return labels


def chexpert_load_imgpath_w_labels(labels_path, base_path):
    labels = chexpert_load_labels(labels_path)
    return [[base_path + img_path, label['label'], label['extra']] for img_path, label in labels.items()]  # it's a path !


def padchest_load_labels(path):
    labels_df = pd.read_csv(path, index_col=False)
    img_name_col = labels_df['ImageID']
    label_col = labels_df['Labels']
    label_col = label_col.str.strip('[]')
    label_col = label_col.swifter.apply(lambda x: str(x).replace("'", '').split(', '))
    sex_col = convert_sex_to_bin(labels_df['PatientSex_DICOM'])

    labels_df['PatientBirth'] = labels_df['PatientBirth'].fillna(labels_df['PatientBirth'].mean())  # avg age for nans
    extra = [{'age': 2013 - int(birth), 'sex': sex, 'position': pos}
             for birth, sex, pos in zip(labels_df['PatientBirth'], sex_col, labels_df['Projection'])]

    labels = {}
    for name, label, e in zip(img_name_col, label_col, extra):
        label_mapped = [label_mapping(j) for j in label]
        labels[name] = {'label': label_mapped, 'extra': e}
    return labels


def padchest_load_imgpath_w_labels(labels_path, img_path):
    labels = padchest_load_labels(labels_path)
    img_paths = [img_path for img_path in glob(img_path + '*.png')]
    return [[img, labels[os.path.basename(img)]['label'], labels[os.path.basename(img)]['extra']] for img in img_paths]


def moco_load_labels(path):
    sex_col = []
    age_col = []
    label_col = []
    files = glob(path + '/*.txt')
    txts = [os.path.basename(file_path) for file_path in files]
    for txt in txts:
        if txt.endswith('_0.txt'):
            label_col.append('normal')
        else:
            label_col.append('tuberculosis')
    for data_file in files:
        with open(data_file, "r") as f:
            for line in f:
                if "Age" in line:
                    age_col.append(line)
                elif "Sex" in line:
                    sex_col.append(line)

    age_col = [i.replace("Patient's Age:", '').replace('yr', '').replace('s', '') for i in age_col]
    sex_col = [s.replace("Patient's Sex:", '').strip() for s in sex_col]
    sex_col = convert_sex_to_bin(sex_col)

    extra = [{'age': age, 'sex': sex} for age, sex in zip(age_col, sex_col)]
    labels = {}
    for txt, label, e in zip(txts, label_col, extra):
        labels[txt[:-4]] = {'label': [label_mapping(label)], 'extra': e}  # remove .txt, so we can match with image name
    return labels


def moco_load_imgpath_w_labels(labels_path, img_path):
    labels = moco_load_labels(labels_path)
    img_paths = [img_path for img_path in glob(img_path + '*.png')]  # note: ugly
    return [[img, labels[os.path.basename(img)[:-4]]['label'], labels[os.path.basename(img)[:-4]]['extra']] for img in img_paths]


def shenzhen_load_labels(path):
    sex_col = []
    age_col = []
    label_col = []
    txts = glob(path + '/*.txt')
    img_name_col = [os.path.basename(file_path) for file_path in txts]
    for name in img_name_col:
        if name.endswith('_0.txt'):
            label_col.append('normal')
        else:
            label_col.append('tuberculosis')

    for txt in txts:
        with open(txt, "r") as f:
            first_line = f.readline()
            first_line = first_line.replace(',', '').replace(' ', '').replace('\t', '').replace('\n', '').lower()
            split_n = 4 if first_line[0] == 'm' else 6
            sex = first_line[:split_n]
            age = first_line[split_n:]
            if 'yr' in age:
                age = int(age.replace('yr', '').replace('s', ''))
            elif 'month' in age:
                age = int(age.replace('month', '').replace('s', '')) / 12.
            elif 'day' in age:
                age = int(age.replace('day', '').replace('s', '')) / 365.
            else:
                age = int(age)
            sex_col.append(sex)
            age_col.append(age)
        sex_col = convert_sex_to_bin(sex_col)

    extra = [{'age': age, 'sex': sex} for age, sex in zip(age_col, sex_col)]
    labels = {}
    for name, label, e in zip(img_name_col, label_col, extra):
        label = [label_mapping(label)]
        labels[name[:-4]] = {'label': label, 'extra': e}

    return labels


def shenzhen_load_imgpath_w_labels(labels_path, img_path):
    labels = shenzhen_load_labels(labels_path)
    img_paths = [img_path for img_path in glob(img_path + '*.png')]
    return [[img, labels[os.path.basename(img)[:-4]]['label'], labels[os.path.basename(img)[:-4]]['extra']] for img in img_paths]


def indianaU_load_labels(path):  # FIXME OUT OF ORDER
    files = glob(path + 'reports/*.xml')
    label_col = []
    img_name_col = []
    pos_col = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            tree = objectify.parse(f)
            root = tree.getroot()
            for i in root.MeSH.major:
                if i == 'normal':  # only normal, only frontal=first in the order
                    ids = sorted([p.attrib['id'] for p in root.findall('parentImage')])
                    if len(ids) > 0:
                        img_name_col.append(ids[0] + '.png')  # use only first image
                    pos_col.append('Frontal')

    '''ms = [str(m).lower() for m in root.MeSH.findall('major')]
                ms13 = []
                for m in ms:
                    if '/' in m:
                        ms13.append(m[m.index('/')])
                    else:
                        ms13.append(m)
                label_col.extend(ms13)'''
            #print(label_col)

    '''unique_labels = set(label_col) - set(k.lower() for k in LABEL_NORM_MAP)
    print(sorted(unique_labels))
    print(len(unique_labels))'''

    extra = [{'position': pos_col[i]} for i in range(len(pos_col))]
    labels = {}
    for name, label, e in zip(img_name_col, label_col, extra):
        label = [label_mapping(label)]
        labels[name] = {'label': label, 'extra': e}
    return labels


def mendeley_load_labels(path):  # and img path
    label_col = []
    pneumonia_type = []
    normal_train_files = glob(path + 'train/NORMAL/*.jpeg')
    normal_test_files = glob(path + 'test/NORMAL/*.jpeg')
    normal_files = normal_train_files + normal_test_files
    pneumonia_train_files = glob(path + 'train/PNEUMONIA/*.jpeg')
    pneumonia_test_files = glob(path + 'test/PNEUMONIA/*.jpeg')
    pneumonia_files = pneumonia_train_files + pneumonia_test_files
    images = [file_path for file_path in normal_files]
    images.extend([file_path for file_path in pneumonia_files])
    for img_path in images:
        if 'VIRUS-' in img_path:
            label_col.append('pneumonia')  # TODO separate label
            pneumonia_type.append('viral')
        elif 'BACTERIA-' in img_path:
            label_col.append('pneumonia')
            pneumonia_type.append('bacterial')
        else:
            label_col.append('normal')
            pneumonia_type.append('none')

    extra = [{'type': pneumonia_type[i]} for i in range(len(pneumonia_type))]
    labels = {}
    for img_path, label, e in zip(images, label_col, extra):
        label = [label_mapping(label)]
        labels[img_path] = {'label': label, 'extra': e}
    return labels


def mendeley_load_imgpath_w_labels(labels_path, base_path):
    labels = mendeley_load_labels(base_path)
    return [[img_path, label['label'], label['extra']] for img_path, label in labels.items()]  # it's a path !


# assembles all the datasets to load them together
def assemble_datasets(dataset_loaders):
    assembled = []
    for name, (loader, lab_path, img_path) in dataset_loaders.items():
        img_w_labels = loader(lab_path, img_path)
        # add dataset name as extra
        for path, label, extra in img_w_labels:
            extra['dataset'] = name
        assembled += img_w_labels
        print('DATASET APPENDED: {}'.format(name))
    return assembled


# filters out irrelevant labels
def filter_assembled(assembled):
    # remove labels of -1 and remove samples where -1 was the only label
    for ass in assembled:
        ass[1] = [label for label in ass[1] if label != -1]
    assembled = [ass for ass in assembled if len(ass[1]) > 0]
    return assembled


# converts labels to onehot encodings
def onehot(indices, length):
    a = np.array([0] * length)
    a[indices] = 1
    return a


# loads image, augments it randomly, yields original and augmented
# takes images with labels lists (including all selected datasets)
class DataGenFile:
    def __init__(self, imgs_with_labels, aug_funcs, norm_func, nlabels, img_dim, augment=True, classes_to_augment=[]):
        self.imgs_with_labels = imgs_with_labels
        self.aug_funcs = aug_funcs
        self.norm_func = norm_func
        self.nlabels = nlabels
        self.img_dim = img_dim
        self.augment = augment
        if augment:
            self.rand_pool = np.random.random((len(imgs_with_labels), len(aug_funcs)))
            self.classes_to_augment = set([LABEL_NORM_MAP[c.lower().replace(' ', '')] for c in classes_to_augment])
        self.images_skipped = []

    def __call__(self):  # run after -1s are filtered out
        for i, (img_path, label, _) in enumerate(self.imgs_with_labels):
            img = cv2.imread(img_path)  # very fast
            if img is None:
                self.images_skipped.append(img_path)
                continue
            img = resize(img, img_dim)  # uint8 -> float64
            if len(img.shape) == 3:
                img = np.mean(img, -1)

            # augmentation
            if self.augment and self.classes_to_augment.intersection(label):
                aug_img, actually_aug = augment_img(img, self.aug_funcs, self.rand_pool[i, :])
                if actually_aug:
                    yield self.norm_func(aug_img).astype(np.float32), onehot(label, self.nlabels).astype(np.float32), True

            yield self.norm_func(img).astype(np.float32), onehot(label, self.nlabels).astype(np.float32), False


def accumulate_in_hdf5(assembled, h5file_path, img_dim, lab_dim, augment, augment_chances, classes_to_augment, valid_ratio):
    # create hdf5 file of preprocessed images for fast loading and training
    compression = tables.Filters(complevel=5, complib='bzip2')
    h5file = tables.open_file(h5file_path, 'w', filters=compression)
    storage_train_x = h5file.create_earray(h5file.root, 'train_x', tables.UInt8Atom(), shape=(0, img_dim[0], img_dim[1]))
    storage_test_x = h5file.create_earray(h5file.root, 'test_x', tables.UInt8Atom(), shape=(0, img_dim[0], img_dim[1]))
    storage_train_y = h5file.create_earray(h5file.root, 'train_y', tables.UInt8Atom(), shape=(0, lab_dim))
    storage_test_y = h5file.create_earray(h5file.root, 'test_y', tables.UInt8Atom(), shape=(0, lab_dim))

    images = DataGenFile(assembled, augment_chances, to_uint8, lab_dim, img_dim, augment, classes_to_augment)
    for i, (img, lab, is_augm) in enumerate(images()):
        lab = lab.astype(np.uint8)
        if np.random.random() < valid_ratio and not is_augm:
            storage_test_x.append(img[None])
            storage_test_y.append(lab[None])
        else:
            storage_train_x.append(img[None])
            storage_train_y.append(lab[None])

        if i % 1000 == 0:
            print('{}/lot is done, where lot > {}'.format(i, len(assembled)))

    h5file.close()
    print('HDF5 FILE SAVED')

    print('IMAGES SKIPPED:', file=sys.stderr)
    print(images.images_skipped, file=sys.stderr)


def h5file_name(nimages, img_dim, lab_dim, augment, augment_chances, valid_ratio):
    return '{}k_{}valid_{}x{}_{}lab_xr14.h5'.format(round(nimages // 1000) if nimages > 0 else 'all', valid_ratio,
                                                    img_dim[0], img_dim[1], lab_dim)  # FIXME 14 as input


class DataGenH5:
    def __init__(self, h5file_path, aug_funcs, norm_func, training, augment=False, dtype=np.float32):
        self.h5file_path = h5file_path
        self.aug_funcs = aug_funcs
        self.norm_func = norm_func
        self.augment = augment
        self.training = training
        self.dtype = dtype
        # can't use rand pool, no idea how many images we have

    def __call__(self):  # images are already resized, labels are onehot
        with tables.open_file(self.h5file_path, 'r') as h5file:
            if self.training:
                images = h5file.root.train_x
                labels = h5file.root.train_y
            else:
                images = h5file.root.test_x
                labels = h5file.root.test_y

            # i = 0  # FIXME rm
            for img, label in zip(images, labels):
                yield self.norm_func(img).astype(np.float32), label.astype(np.float32)

                # augmentation
                if self.augment:
                    aug_img, actually_aug = augment_img(img, self.aug_funcs, np.random.random(len(self.aug_funcs)))
                    if actually_aug:
                        yield self.norm_func(aug_img).astype(self.dtype), label.astype(self.dtype)
                # if i > 128:
                #     break
                # i += 1

    def __len__(self):
        with tables.open_file(self.h5file_path, 'r') as h5file:
            return len(h5file.root.train_y)


# mendeley_load_labels('/mnt/data/pneumonia/mendeley')
# padchest_load_labels('/mnt/data/pneumonia/padchest/chest_x_ray_images_labels.csv')
# indianaU_load_labels('/mnt/data/pneumonia/indiana/')
# cxr14_load_labels('/mnt/data/pneumonia/chestxray14/Data_Entry_2017.csv')
# shenzhen_load_labels('/mnt/data/pneumonia/shenzen/ClinicalReadings')
# cheXpert_load_labels('/mnt/data/pneumonia/mimic')
# pprint([j for i,j in enumerate(cheXpert_load_labels('/mnt/data/pneumonia/chexpert/').items()) if i < 10])

BF = '/mnt/data/pneumonia/'
DATASET_LOADERS = {
    'cxr14':    (cxr14_load_imgpath_w_labels, BF + 'cxr14/Data_Entry_2017.csv', BF + 'cxr14/'),
    'chexpert': (chexpert_load_imgpath_w_labels, BF + 'chexpert/', BF + 'chexpert/'),
    'mimiccxr': (chexpert_load_imgpath_w_labels, BF + 'mimic/', BF + 'mimic/'),
    'padchest': (padchest_load_imgpath_w_labels, BF + 'padchest/chest_x_ray_images_labels.csv', BF + 'padchest/images/*/'),
    'moco':     (moco_load_imgpath_w_labels, BF + 'montgomery/labels/', BF + 'montgomery/images/'),
    'shenzhen': (shenzhen_load_imgpath_w_labels, BF + 'shenzhen/labels', BF + 'shenzhen/images/'),
    'mendeley': (mendeley_load_imgpath_w_labels, BF + '', BF + 'mendeley/chest_xray/')
}

# TODO define augmentation for all classes, but with lower chances (we want augmentation
# TODO make data generation parallel

if __name__ == '__main__':

    class Stage(Enum):
        ASSEMBLE = 0
        SAVE_TO_HDF5 = 1
        ANAL = 2

    # config
    stage = Stage.ANAL
    nimages = 100000  # to save in hdf5; -1 == all
    base_dir = '/mnt/data/pneumonia/'
    img_dim = (512, 512)
    lab_dim = len(LABEL_NORM_MAP)
    augment_b4_h5 = True  # augment and then save to hdf5
    augment_b4_training = False  # xor augment_b4_h5 (don't set this to true pls)
    augment_chances = {stretch_image: 0.2, flip_image: 0.5, rotate_image: 0.3}
    classes_to_augment = ['pneumonia']  # TODO add more relevant
    valid_ratio = 0.2

    # assemble and save temporal file
    if stage.value == Stage.ASSEMBLE.value:
        assembled = assemble_datasets(DATASET_LOADERS)
        print('LEFT OUT:', file=sys.stderr, flush=True)
        pprint(sorted(list(LEFT_OUT)))
        with open(base_dir + 'assembled.pickle', 'wb') as f:
            pickle.dump(assembled, f)

    # load images, augment them, and save to hdf5
    if stage.value == Stage.SAVE_TO_HDF5.value:
        with open(base_dir + 'assembled.pickle', 'rb') as f:
            assembled = pickle.load(f)

        assembled = filter_assembled(assembled)
        print('SAMPLE SIZE:', len(assembled))

        random.shuffle(assembled)
        if nimages > 0:
            assembled = assembled[:nimages]

        h5file_path = base_dir + h5file_name(nimages, img_dim, lab_dim, augment_b4_h5, augment_chances, valid_ratio)
        accumulate_in_hdf5(assembled, h5file_path, img_dim, lab_dim, augment_b4_h5, augment_chances, classes_to_augment, valid_ratio)

    # load assembled and plot statistics
    if stage.value == Stage.ANAL.value:
        with open(base_dir + 'assembled.pickle', 'rb') as f:
            assembled = pickle.load(f)

        assembled = filter_assembled(assembled)
        print('SAMPLE SIZE:', len(assembled))

        # label count
        labels = np.array([l for a in assembled for l in a[1]])
        count = {}
        for l in labels:
            if l not in count:
                count[l] = 0
            count[l] += 1
        count = {LABEL_REV_MAP[k]: v for k, v in count.items()}
        pprint(count)

        # plot figures of certain class
        # plt.bar(range(len(count)), list(count.values()), align='center')
        # plt.xticks(range(len(count)), list(count.keys()))
        # plt.show()

        # sex count
        sex = np.array([a[2]['sex'] for a in assembled if 'sex' in a[2]])
        nfem = np.sum(sex == 0)
        nmal = np.sum(sex == 1)
        print(nfem, nmal)
        plt.bar(range(2), [nfem, nmal], align='center')
        plt.xticks(range(2), ['Female', 'Male'])
        plt.show()

        # img size count
        img = np.array([a[0] for a in assembled])
        size = []
        for i in img:
            im = cv2.imread(i)
            size.append(im.shape)

        width = [i[0] for i in size]
        height = [i[1] for i in size]
        width_count = collections.Counter(width)
        plt.bar(range(len(width_count)), list(width_count.values()), align='center', label='width')
        height_count = collections.Counter(height)
        plt.bar(range(len(height_count)), list(height_count.values()), align='center', label='height')
        plt.legend()
        plt.show()


# IMAGES SKIPPED: ['/mnt/data/pneumonia/padchest/images/46/216840111366964012558082906712009327122220177_00-102-064.png', '/mnt/data/pneumonia/padchest/images/43/216840111366964012339356563862009072111404053_00-043-192.png', '/mnt/data/pneumonia/padchest/images/13/216840111366964013590140476722013058110301622_02-056-111.png', '/mnt/data/pneumonia/padchest/images/48/216840111366964012819207061112010306085429121_04-020-102.png', '/mnt/data/pneumonia/padchest/images/44/216840111366964012373310883942009170084120009_00-097-074.png', '/mnt/data/pneumonia/padchest/images/43/216840111366964012487858717522009280135853083_00-075-001.png', '/mnt/data/pneumonia/padchest/images/44/216840111366964012819207061112010281134410801_00-129-131.png', '/mnt/data/pneumonia/padchest/images/42/216840111366964012373310883942009117084022290_00-064-025.png', '/mnt/data/pneumonia/padchest/images/43/216840111366964012558082906712009301143450268_00-075-157.png', '/mnt/data/pneumonia/padchest/images/43/216840111366964012989926673512011101154138555_00-191-086.png', '/mnt/data/pneumonia/padchest/images/49/216840111366964012819207061112010307142602253_04-014-084.png', '/mnt/data/pneumonia/padchest/images/41/216840111366964012989926673512011151082430686_00-157-045.png', '/mnt/data/pneumonia/padchest/images/43/216840111366964012283393834152009033102258826_00-059-087.png', '/mnt/data/pneumonia/padchest/images/46/216840111366964012989926673512011074122523403_00-163-058.png', '/mnt/data/pneumonia/padchest/images/41/216840111366964012989926673512011132200139442_00-157-099.png', '/mnt/data/pneumonia/padchest/images/45/216840111366964012339356563862009068084200743_00-045-105.png', '/mnt/data/pneumonia/padchest/images/46/216840111366964012989926673512011083134050913_00-168-009.png', '/mnt/data/pneumonia/padchest/images/17/216840111366964013590140476722013043111952381_02-065-198.png', '/mnt/data/pneumonia/padchest/images/49/216840111366964012819207061112010315104455352_04-024-184.png', '/mnt/data/pneumonia/padchest/images/42/216840111366964013076187734852011291090445391_00-196-188.png', '/mnt/data/pneumonia/padchest/images/45/216840111366964012558082906712009300162151055_00-078-079.png', '/mnt/data/pneumonia/padchest/images/16/216840111366964013649110343042013092101343018_02-075-146.png', '/mnt/data/pneumonia/padchest/images/17/216840111366964013590140476722013049100117076_02-063-097.png', '/mnt/data/pneumonia/padchest/images/46/216840111366964012373310883942009152114636712_00-102-045.png', '/mnt/data/pneumonia/padchest/images/19/216840111366964013829543166512013353113303615_02-092-190.png', '/mnt/data/pneumonia/padchest/images/44/216840111366964012373310883942009180082307973_00-097-011.png']
