import os
import nltk
import pickle
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

nltk.download('punkt')


# step 1: build vocab ----------------------------------------------

class Vocab:
    """Simple vocabulary wrapper"""

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0

    def __call__(self, token):
        if token not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]

    def __len__(self):
        return len(self.w2i)

    def add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


def build_vocabulary(json, threshold):
    """Build a simple vocabulary wrapper"""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print(f'[{i + 1}/{len(ids)}] Tokenized the captions.')

    # If the word frequency is less than 'threshold', then the word is discarded
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens
    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    # Add the words to the vocabulary
    for i, token in enumerate(tokens):
        vocab.add_token(token)
    return vocab


vocab = build_vocabulary(
    json='/media/tyler/Elements/unzipped-image-datasets/cocodataset/annotations/captions_train2014.json', threshold=4)
vocab_path = '/media/tyler/Elements/unzipped-image-datasets/cocodataset/vocabulary.pkl'
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print(f'Total vocabulary size: {len(vocab)}')
print(f'Saved the vocabulary wrapper to \'{vocab_path}\'')


# step 2: resize images ----------------------------------------------
def reshape_image(image, shape):
    """Resize an image to the given shape"""
    return image.resize(shape, Image.ANTIALIAS)


def reshape_images(image_path, output_path, shape):
    """Reshape the images in 'image_path' and save into 'output_path'. """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = os.listdir(image_path)
    num_im = len(images)
    for i, im in enumerate(images):
        with open(os.path.join(image_path, im), 'r+b') as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_path, im), image.format)
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{num_im}] Resized the images and saved into '{output_path}'")


image_path = '/media/tyler/Elements/unzipped-image-datasets/cocodataset/train2014/'
output_path = '/media/tyler/Elements/unzipped-image-datasets/cocodataset/resized_images/'
image_shape = [256, 256]
reshape_images(image_path, output_path, image_shape)


# step 3: instantiate data loader ----------------------------------------------
