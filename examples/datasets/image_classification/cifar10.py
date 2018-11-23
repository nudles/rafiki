#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()


import tarfile
import os
import sys
import tempfile
import numpy as np
import csv
import shutil
from tqdm import tqdm
from PIL import Image
try:
    import pickle
except ImportError:
    import cPickle as pickle

from rafiki.model import ModelUtils


def write_dataset(images, labels, out_dataset_path):
    with tempfile.TemporaryDirectory() as d:
        # Create images.csv in temp dir for dataset
        # For each (image, label), save image as .png and add row to images.csv
        # Show a progress bar in the meantime
        images_csv_path = os.path.join(d, 'images.csv')
        n = len(images)
        with open(images_csv_path, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'class'])
            writer.writeheader()
            for (i, image, label) in tqdm(zip(range(n), images, labels), total=n, unit='images'):
                image_name = '{}-{}.png'.format(label, i)
                image_path = os.path.join(d, image_name)
                image = image.transpose(1, 2, 0)
                pil_image = Image.fromarray(image, mode='RGB')
                pil_image.save(image_path)
                writer.writerow({'path': image_name, 'class': label})

        # Zip and export folder as dataset
        out_path = shutil.make_archive(out_dataset_path, 'zip', d)
        # Remove additional trailing `.zip`
        os.rename(out_path, out_dataset_path)


def extract_tarfile(filepath):
    if os.path.exists(filepath):
        print('The tar file does exist. Extracting it now..')
        with tarfile.open(filepath, 'r') as f:
            f.extractall('.')
        print('Finished!')
        sys.exit(0)


def check_dir_exist(dirpath):
    if os.path.exists(dirpath):
        print('Directory %s does exist. To redownload the files, '
              'remove the existing directory and %s.tar.gz' % (dirpath, dirpath))
        return True
    else:
        return False


def do_download(dirpath, url):
    if check_dir_exist(dirpath):
        sys.exit(0)
    print('Downloading CIFAR10 from %s' % (url))
    # urllib.request.urlretrieve(url, gzfile)
    utils = ModelUtils()
    gzfile = utils.download_dataset_from_uri(url)
    extract_tarfile(gzfile)
    print('Finished!')


def load_dataset(filepath):
    print('Loading data file %s' % filepath)
    with open(filepath, 'rb') as fd:
        try:
            cifar10 = pickle.load(fd, encoding='latin1')
        except TypeError:
            cifar10 = pickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)


if __name__ == '__main__':
    dirpath = 'cifar-10-batches-py'
    # gzfile = 'cifar-10-python' + '.tar.gz'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    do_download(dirpath, url)
    train_images, train_labels = load_train_data(dirpath)
    test_images, test_labels = load_test_data(dirpath)

    print('Converting and writing datasets...')

    write_dataset(train_images, train_labels,
                  label_to_index, out_train_dataset_path)
    print('Train dataset file is saved at {}'.format(out_train_dataset_path))

    write_dataset(test_images, test_labels,
                  label_to_index, out_test_dataset_path)
    print('Test dataset file is saved at {}'.format(out_test_dataset_path))
