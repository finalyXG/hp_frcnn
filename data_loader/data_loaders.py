from pathlib import Path
from random import Random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import utils.pytorch_vision.transforms as T
from base import BaseDataLoader

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], sum_partition_size=None, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        if sum_partition_size is not None:
            data_len = int(data_len / sum_partition_size) * sum_partition_size
        # data_len = sum([int(data_len * s) for s in sizes]) # Laurence 20230611
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    

    

class BboxDataset(Dataset):
    def __init__(self, ds, ds_seg, train_size, ds_name,cls_ls, cls_ls_old=[], cls_map=None, is_train=True):
        super().__init__()
        self.ds = ds
        self.ds_seg = ds_seg
        self.is_train = is_train
        self.train_size = train_size
        self.transforms = self.get_transform()
        self.ds_name = ds_name
        self.cls_map = cls_map
        self.cls_ls_old = cls_ls_old
        self.cls_ls_new = cls_ls
        self.cls_ls = cls_ls
        self.cls_dict = dict(zip(range(len(self.cls_ls_new)), self.cls_ls_new))
        self.cls_dict_len = len(self.cls_dict)
        self.cls_old_num = len(self.cls_ls_old)
        self.cls_new_num = len(self.cls_ls_new)
        
    def get_transform(self):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        # if self.is_train:
        #     transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
            

    def __len__(self):
        # return 600
        return len(self.ds)
    
    def get_seg_map_bbox(self, seg, kernel = np.ones((10, 10),np.uint8)):
        bbox_dict = {i+1: [] for i in range(self.cls_old_num)}
        # if self.ds_name == 'cihp':
        #     seg = np.where(seg == 10, 13, seg)
            
        print(444/0)
        for i in range(self.cls_old_num):
            cls_ind = i + 1
            # Perform erosion
            eroded_img = (seg == cls_ind).astype(np.uint8)
            # eroded_img = cv2.erode( eroded_img, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find the bounding box of each contour
            bounding_boxes = [list(cv2.boundingRect(cnt)) for cnt in contours]
            bboxes = bounding_boxes
            bboxes = [b for b in bboxes if b[2] > 5 and b[3] > 5 ] # check if w and h of a box are larger than a threshold 
            
            cond_lip  = self.ds_name == 'lip' and cls_ind in  [1,2,5,6,7,9,10,11,12,13,14,15,16,17,18,19]
            cond_cihp = self.ds_name == 'cihp' and cls_ind in [1,2,5,6,7,9,10,11,12,13,14,15,16,17,18,19]
            if len(bboxes) > 1 and (cond_lip or cond_cihp):
                # Find the minimum and maximum x and y coordinates
                min_x = min([box[0] for box in bboxes])
                min_y = min([box[1] for box in bboxes])
                max_x = max([box[0] + box[2] for box in bboxes])
                max_y = max([box[1] + box[3] for box in bboxes])

                # Create a new bounding box that encompasses all of the original bounding boxes
                bboxes = [[min_x, min_y, max_x - min_x, max_y - min_y]]
            
            bbox_dict[cls_ind] = bboxes
        
        bbox_dict_new = {i+1: [] for i in range(self.cls_new_num)}
        for cls_ind, item in bbox_dict.items():
            bbox_dict_new[self.cls_map[cls_ind]].extend(bbox_dict[cls_ind])
            
        return bbox_dict_new
    
    def resize_with_max_size(self, im, max_size):
        # Open the image file
        # Get the original width and height
        width, height = im.size

        # Determine the aspect ratio of the image
        aspect_ratio = width / height

        # Calculate the new width and height based on the maximum size
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        im = im.resize((new_width, new_height))

        return im
    
    def show_sample_img_w_bbox(self, idx, save_path=None, plt_show=True, return_fig=False, max_size=600):
        # Draw the bounding boxes on the image
        fig, ax = plt.subplots()
        sample = self.__getitem__(idx)
        img = sample[0].permute(1,2,0).numpy()
        labels= sample[1]['labels'].numpy()
        bboxes = sample[1]['boxes'].numpy()
        ax.imshow(img)
        for bbox, lbl in zip(bboxes,labels):
            x, y, x1, y1 = bbox
            w = x1 - x
            h = y1 - y
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            plt.text(x, y, self.cls_dict[lbl], fontsize=12, color='red', ha='left', va='top') # + "-" + str(lbl)
            
            ax.add_patch(rect)
        if save_path is not None:
            plt.savefig(save_path)
        if return_fig:
            # render the figure to a buffer
            fig.set_dpi(300)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            w, h = pil_image.size
            pil_image = pil_image.resize((int(w / 4), int(h / 4) ))            
            pil_image = self.resize_with_max_size(pil_image, max_size)
            return pil_image
        if plt_show:
            plt.show()
            
        
    # def __getitem__(self, idx):
    def __getitem__gen(self, idx):
        p = Path(self.ds[idx]['image'].filename.replace(f"{self.ds_name}_img",f"{self.ds_name}_bbox").replace(".jpg",".npy"))
        bbox_dict = {}
        if not p.exists():
            seg = np.array(self.ds_seg[idx]['image'])  
            bbox_dict = self.get_seg_map_bbox(seg)
            np.save(str(p), np.array(bbox_dict))
            
        return 0
    
    # def __getitem__new(self, idx):
    def __getitem__(self, idx):
        p = Path(self.ds[idx]['image'].filename.replace(f"{self.ds_name}_img",f"{self.ds_name}_bbox").replace(".jpg",".npy"))
        bbox_dict = {}
        assert p.exists(), f"Can not find: {p}"
        bbox_dict = np.load(p, allow_pickle=True).item()
        img = self.ds[idx]['image']
        img_id = idx
        if not self.is_train:
            img_id += self.train_size + 1
        target = {'boxes':[], 'labels':[], 'image_id': torch.tensor([img_id]), 'area':[], 'iscrowd':[]}
        for k,item in bbox_dict.items():
            if len(item) > 0:
                for box in item:
                    x, y, w, h = box
                    target['boxes'].append([x,y,x+w,y+h])
                    target['labels'].append(k)
                    target['area'].append(w * h)
                    target['iscrowd'].append(0)
                    
        if len(target['boxes']) == 0:
            # assert self.is_train
            return self.__getitem__(0)
        
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        # label mapping
        if self.cls_map is not None:
           target['labels'] = [self.cls_map[e] for e in target['labels']]
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.int64)
        

            
        assert torch.all(target['labels'] < self.cls_dict_len) 
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    
    
    
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
