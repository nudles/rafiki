#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# the code is modified from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from singa import autograd
from singa import tensor
from singa import device
from singa import opt

import numpy as np
from tqdm import trange


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return autograd.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, 'int').sum() / float(len(t))


class BasicBlock(autograd.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class Bottleneck(autograd.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = autograd.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = autograd.Conv2d(planes, planes, kernel_size=3,
                                     stride=stride,
                                     padding=1, bias=False)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.conv3 = autograd.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = autograd.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = autograd.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class ResNet(autograd.Layer):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = autograd.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = autograd.BatchNorm2d(64)
        self.maxpool = autograd.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = autograd.AvgPool2d(7, stride=1)
        self.fc = autograd.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv = autograd.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            bn = autograd.BatchNorm2d(planes * block.expansion)

            def downsample(x):
                return bn(conv(x))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        def forward(x):
            for layer in layers:
                x = layer(x)
            return x
        return forward

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = autograd.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = autograd.flatten(x)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model

from sklearn import svm
import json
import pickle
import os
import base64
import numpy as np

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class
from rafiki.constants import TaskType, ModelDependency


class SingaResNet(BaseModel):
    '''
    Implements ResNet using Singa for simple image classification
    '''

    def get_knob_config(self):
        return {
            'knobs': {
                'max_epoch': {
                    'type': 'int',
                    'range': [10, 10]
                },
                'lr': {
                    'type': 'float_exp',
                    'range': [1e-1, 1e-4]
                }
                'batch_size': {
                    'type': 'int_cat',
                    'range': [16, 32, 64, 128, 256]
                }
                'weight_decay': {
                    'type': 'float_exp',
                    'range': '1e-3, 1e-5'
                }
                'momentum': {
                    'type': 'float',
                    'range': [0.5, 0.9]
                }
            }
        }

    def init(self, knobs):
        self._max_epoch = knobs.get('max_epoch')
        self._lr = knobs.get('lr')
        self._batch_size = knobs.get('batch_size')
        self._weight_decay = knobs.get('weight_decay')
        self._momentum = knobs.get('momentum')
        self._clf = resnet18()  # TODO(wangwei) make it an hyperparameter
        self.dev = device.create_cuda_gpu_on(0)

    def train(self, dataset_uri):
        dataset = self.utils.load_dataset_of_image_files(dataset_uri)
        # TODO(wangwei) load images in batch
        (images, classes) = zip(*[(image, image_class)
                                  for (image, image_class) in dataset])
        X = np.array(self._prepare_X(images),
                     dtype=np.float32).transpose(0, 1, 2)
        y = np.array(classes, dtype=np.int)

        self.utils.log('Start intialization............')
        bs = self._batch_size
        sgd = opt.SGD(lr=self._lr, momentum=self._momentum,
                      weight_decay=self._weight_decay)

        c, h, w = X.shape[1:]
        tx = tensor.Tensor((bs, c, h, w), self.dev)
        ty = tensor.Tensor((bs,), self.dev, tensor.int32)
        autograd.training = True
        self.utils.define_loss_plot()
        self.utils.log('Start training............')
        for epoch in range(max_epoch):
            niters = X.shape[0] // bs
            pacc, ploss = 0.0, 0.0
            for b in range(niters):
                tx.copy_from_numpy(X[niters * bs: niters * bs + bs])
                ty.copy_from_numpy(y[niters * bs:niters * bs + bs])
                p = model(tx)
                loss = autograd.softmax_cross_entropy(p, ty)
                for p, g in autograd.backward(loss):
                    # print(p.shape, g.shape)
                    sgd.update(p, g)
                ploss = ploss * 0.9 + loss.to_numpy().average() * 0.1
                # p.to_numpy must be after loss.to_numpy
                acc = accuracy(p.to_numpy(), y[niters * bs:niters * bs + bs])
                pacc = pacc * 0.9 + acc * 0. 1
            # self.utils.log('Train loss: {}'.format(accum_loss))
            self.utils.log_loss_metric(accum_loss)
        self.utils.log('Finish training............')

    def evaluate(self, dataset_uri):
        dataset = self.utils.load_dataset_of_image_files(dataset_uri)
        (images, classes) = zip(*[(image, image_class)
                                  for (image, image_class) in dataset])

        dev = device.create_cuda_gpu_on(0)
        #dev = device.create_cuda_gpu()
        bs = self._batch_size

        c, h, w = X.shape[1:]
        tx = tensor.Tensor((bs, c, h, w), self.dev)
        ty = tensor.Tensor((bs,), self.dev, tensor.int32)
        autograd.training = False

        niters = X.shape[0] // bs
        ploss, pacc = 0.0, 0.0
        for b in range(niters):
            tx.copy_from_numpy(X[niters * bs: niters * bs + bs])
            ty.copy_from_numpy(y[niters * bs:niters * bs + bs])
            p = model(tx)
            loss = autograd.softmax_cross_entropy(p, ty)

            ploss += loss.to_numpy().average()
            pacc += accuracy(p.to_numpy, y[niters * bs:niters * bs + bs])
        # self.utils.log('Evaluation loss: {}'.format(accum_loss))
        self.utils.log_loss_metric(ploss / niters)
        return pacc / niters

    def predict(self, queries):
        X = self._prepare_X(queries)
        tx = tensor.Tensor(X.shape, self.dev)
        autograd.training = False
        tx.copy_from_numpy(X[niters * bs: niters * bs + bs])
        probs = model(tx)
        return probs.to_numpy().tolist()

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Save model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64

        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params.get('clf_base64', None)
        if clf_base64 is None:
            raise InvalidModelParamsException()

        clf_bytes = base64.b64decode(params['clf_base64'].encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

    def _prepare_X(self, images):
        return [np.asarray(image).flatten() for image in images]


if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='SkSvm',
        task=TaskType.IMAGE_CLASSIFICATION,
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
        test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
    )


if __name__ == '__main__':
