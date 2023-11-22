import os
import torch, torchvision
from torchvision.io import read_image, ImageReadMode
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xml.etree.ElementTree as et

class ImageDataset:
    def __init__(self, 
              root, 
              transform=None):
        self.root = root
        self.transform = transform
        
        self.imgs = absoluteFilePaths(os.path.join(root, "Images"))
        self.bboxes = absoluteFilePaths(os.path.join(root, "Annotation"))

        self.classes = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """returns tuple containing label and box coords"""
        image = torchvision.io.read_image(self.imgs[idx], mode = ImageReadMode.RGB)

        root = et.parse(self.bboxes[idx]).getroot() # get the root of the xml
        boxes = list()
        
        for box in root.findall('.//object'):
            label = re.sub(r'-', '_', box.find('name').text).lower()
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            box_data = np.array([xmin,ymin,xmax,ymax])
        
        sample= {"image": image,
                "bbox": box_data,
                "str_label": label.lower(),
                "idx_label": self.classes.index(label.lower())}

        if self.transform:
            sample['image'] = self.transform(sample['image'].type(torch.float))

        return sample

    def display_images(self, rows, cols, figsize, fig_path= None):
        num_images = cols * rows
        
        samples = random.sample(range(self.__len__()), num_images)
        
        fig, ax = plt.subplots(figsize = figsize)

        for i, _ in enumerate(samples):
            sample = self.__getitem__(samples[i])
            img = sample['image'].permute(1, 2,0)
            
            plt.subplot(rows, cols, i + 1)
            plt.title(label = str(sample['str_label']) + "\n" + str(torchvision.transforms.functional.get_image_size(sample['image'])))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(img)

            ax = plt.gca()
            bbox = patches.Rectangle((sample['bbox'][:2]), 
                        sample['bbox'][2], 
                        sample['bbox'][3],
                        linewidth = 4,
                        edgecolor = 'y', facecolor='none')

            ax.add_patch(bbox)

            if fig_path:
                plt.savefig(fig_path, bbox_inches='tight')

def absoluteFilePaths(directory):
    file_list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_list.append(os.path.abspath(os.path.join(dirpath, f)))
    return sorted(file_list)