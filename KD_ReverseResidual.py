import os
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from models.reverse_Residual_ResNet import reverse_student18
from utils.util import  time_string, convert_secs2time, AverageMeter
from utils.functions import cal_anomaly_maps,cal_loss
from utilsModels import calculScoreAndVisualize
from models.teacherTimm import teacherTimm

class KD_ReverseResidual():
    def __init__(self,data_path="dataset/MVTEC",obj='carpet',save_path='results',vis=False,DG=False):
        self.device='cuda'
        self.data_path = data_path
        self.obj = obj
        self.img_resize = 256
        self.img_cropsize = 256
        self.validation_ratio = 0.3
        self.num_epochs = 100
        self.lr = 0.005  
        self.batch_size = 4 
        self.patienceES=10
        self.vis = vis
        self.save_path = save_path
        self.model_dir = save_path + '/models' + '/' + obj+'_Reverse'
        self.img_dir = save_path + '/imgs' + '/' + obj+'_Reverse'
        self.DG=DG
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        self.load_model()
        self.load_dataset()

        self.optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=0.1, patience=5, verbose=True)


    def load_dataset(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=True, resize=self.img_resize, cropsize=self.img_cropsize,blur=True,vis=self.vis)
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)

    def load_model(self):

        self.model_t=teacherTimm(backbone_name="resnet34",out_indices=[0,1,2,3]).to(self.device)
        self.model_s= reverse_student18(DG=self.DG).to(self.device)


        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()

    def train(self):
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()

        compteurEarlyStop=0
        for epoch in range(1, self.num_epochs+1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs+1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, self.num_epochs, time_string(), need_time))
            losses = AverageMeter()
            for (data, label, _) in tqdm(self.train_loader):
                data = data.to(self.device)

                with torch.set_grad_enabled(True):
                    features_t = self.model_t(data)
                    features_s = self.model_s(features_t)
                    loss = cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

            val_loss = self.val(epoch)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
                compteurEarlyStop=0
            else:
                compteurEarlyStop=compteurEarlyStop+1
                if (compteurEarlyStop==self.patienceES):
                    break
            
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('Training end.')
    
    def val(self, epoch):
        self.model_s.eval()
        losses = AverageMeter()
        for (data, _, _) in tqdm(self.val_loader):
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(features_t)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        self.scheduler.step(losses.avg)
        return losses.avg
    def save_checkpoint(self):
        print('Save model !!!')
        state = {'model':self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_s.pth'))

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, 'model_s.pth'))
        except:
            raise Exception('Check saved model path.')
        self.model_s.load_state_dict(checkpoint['model'])
        self.model_s.eval()
        for param in self.model_s.parameters():
            param.requires_grad = False
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, resize=self.img_resize, cropsize=self.img_cropsize,blur=False,vis=self.vis)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        print('Testing')

        for (data, label,mask) in tqdm(test_loader): 
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.extend(mask.squeeze().cpu().numpy())
            data = data.to(self.device)
            with torch.set_grad_enabled(False):

                score = []
                timeBefore = time.perf_counter()
                features_t = self.model_t(data)
                features_s = self.model_s(features_t)
                timeAfterFeatures = time.perf_counter()
                print("inference : " + str(timeAfterFeatures - timeBefore))
                score = cal_anomaly_maps(features_s, features_t, self.img_cropsize)               
                scores.append(score)
     

        scores = np.asarray(scores)
        calculScoreAndVisualize(self,scores,gt_list,gt_mask_list,test_imgs)





    

