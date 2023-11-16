import os
import torch, torchvision

import numpy as np
import re
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
              root, 
              transforms=None):
        self.root = root
        self.transforms = transforms
        
        #self.images_paths = absoluteFilePaths(self.root)
        self.imgs = absoluteFilePaths(os.path.join(root, "Images"))
        self.bboxes = absoluteFilePaths(os.path.join(root, "Annotation"))

        self.classes = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """returns tuple containing label and box coords"""
        image = torchvision.io.read_image(self.imgs[idx])

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