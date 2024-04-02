import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='dataset/MVTEC', class_name='bottle', is_train=True,
                 resize=224, cropsize=224,blur=False,vis=False): 
        
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.blur=blur
        self.vis=vis
        
        self.x, self.y, self.mask = self.load_modified_dataset_folder()


        self.transform_x = T.Compose([T.Resize(resize),
                                        T.CenterCrop(cropsize),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        

        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        if (self.blur):
            x=TF.gaussian_blur(x,3)
        if y == 0 or self.vis==False:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_modified_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        gt_train_dir=os.path.join(self.dataset_path, self.class_name, 'train_ground_truth')
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith(('.png','.tif'))])
            x.extend(img_fpath_list)
            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            elif(phase=='test'):
                if (self.vis == True):
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]

                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                         for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                else:
                    y.extend([1] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
            else :
                if (self.vis == True):
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_train_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]

                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') 
                                         for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                else:
                    y.extend([1] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)