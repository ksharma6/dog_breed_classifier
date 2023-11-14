import os
import torch, torchvision

import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

class ImageDataset:
    def __init__(self, 
                 root_directory, 
                 transformation = None):
        self.root_directory = root_directory
        self.dataset = torchvision.datasets.ImageFolder(root = root_directory, 
                                                      transform = transformation)
        self.class_labels = list(self.dataset.class_to_idx.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, idx_label, str_label = sample[0], sample[1], self.class_labels[int(sample[1])]
        
        sample = {'image': image,
                  'idx_label': torch.tensor(idx_label),
                  'str_label':str_label
                    }
        
        return sample
    
    def display_images(self, rows, cols, num_images, figsize, fontdict):
        image_idxs = random.sample(range(20_000), num_images)
        
        for i, image_idx in enumerate(image_idxs):
            sample = self.__getitem__(image_idx)
            img = sample['image'].permute(1, 2,0)

            plt.rcParams["figure.figsize"] = figsize
            plt.subplot(rows, cols, i + 1)

            plt.title(label = str(sample['str_label']) + "\n" + str(torchvision.transforms.functional.get_image_size(sample['image'])),
                    fontdict = fontdict)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(img)

class BBDataset:
    def __init__(self, 
              root_directory, 
              transformation=None):
        self.root = root_directory
        self.transform = transformation
        self.subfolders = list(sorted(os.listdir(self.root)))
        self.images_paths = absoluteFilePaths(self.root)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        """returns tuple containing label and box coords"""
        img_path = self.images_paths[idx]

        root = et.parse(img_path).getroot() # get the root of the xml
        boxes = list()
        for box in root.findall('.//object'):
            label = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            data = np.array([xmin,ymin,xmax,ymax])
            boxes.append(data)
        
        bb = tuple([label, data])
        return bb

def absoluteFilePaths(directory):
    file_list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_list.append(os.path.abspath(os.path.join(dirpath, f)))
    return sorted(file_list)