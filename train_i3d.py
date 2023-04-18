import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data.distributed
import sys
#from data_gen.batch_generator import BatchGenerator
import random
from arg_parse import parse_args
# from data_gen.dataloader import *
# from data_gen.preprocessing import DataAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms
from train_step import *
import torch.backends.cudnn as cudnn
import warnings
import numpy as np
import time
import datetime
from MMIR import Model
from pytorch_i3d import InceptionI3d
from torchinfo import summary
from utils import (train, validate, build_dataflow, get_augmentor,
                         save_checkpoint)
from video_dataset import VideoDataSet
import tqdm



class TrainTestScript:

    def __init__(self, FLAGS, results_dir, train_dir):
        # inputs
        self.FLAGS = FLAGS
        self.train_dir = train_dir
        self.datasets = FLAGS.dataset
        self.unseen_dataset = FLAGS.unseen_dataset
        self.num_gpus = FLAGS.num_gpus
        self.num_labels = FLAGS.num_labels
        self.target_data = not (not (FLAGS.domain_mode))

        self.save_model = train_dir
        if FLAGS.restoring:
            self.restoring = True
        else:
            self.restoring = False


        if self.target_data:
            
            if FLAGS.domain_mode == "None" or FLAGS.domain_mode == "Pretrain":
                self.target_data = False
                print("No adaptation")

        if FLAGS.domain_mode:
            self.domain_mode = FLAGS.domain_mode
        else:
            self.domain_mode = "None"

        self.lr = FLAGS.lr

        if not FLAGS.modality:
            raise Exception("Need to Specify modality")

        if FLAGS.modality != "rgb" and FLAGS.modality != "flow" and FLAGS.modality != "joint":
            raise Exception("Invalid Modality")

        self.results_dir = results_dir + "_" + FLAGS.modality
        self.modality = FLAGS.modality

        

        #if self.domain_loss or self.bn_align or self.discrepancy or self.mmd:
        # self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
        #                    temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update, learning_rate=self.lr,
        #                    domain_mode=self.domain_mode,steps_per_update=FLAGS.steps_before_update,
        #                    aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,




    def train(self):

        phase = 'Train'

        dist.init_process_group(backend='nccl')
        #torch.cuda.set_device(args.local_rank)

        if not os.path.exists(self.results_dir) and dist.get_rank() == 0:
            os.makedirs(self.results_dir)

        if not os.path.exists(self.train_dir) and dist.get_rank() == 0:
            os.makedirs(self.train_dir)


        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)


        batch_per_gpu = self.FLAGS.batch_size//self.num_gpus
        print("batch_per_gpu",batch_per_gpu)

        # train_data_S = Epic(self.num_labels, self.datasets,
        #                             temporal_window=self.FLAGS.temporal_window,
        #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                            synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)

        # sampler_S = DistributedSampler(train_data_S)
        # trainloader_s = DataLoader(train_data_S,batch_size=batch_per_gpu,shuffle=False,sampler=sampler_S,num_workers=1,drop_last=True)

        # # val_data_S = Epic(self.num_labels, self.datasets,
        # #                             temporal_window=self.FLAGS.temporal_window,
        # #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        # #                            synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)

        # # #val_sampler_S = DistributedSampler(val_data_S)
        # # valoader_s = DataLoader(val_data_S,batch_size=batch_per_gpu,shuffle=False,num_workers=0,drop_last=True)

        

        # train_data_T = Epic(self.num_labels,self.unseen_dataset,
        #                                   temporal_window=self.FLAGS.temporal_window,
        #                                   rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                                   synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)

        # sampler_T = DistributedSampler(train_data_T)
        # trainloader_t = DataLoader(train_data_T,batch_size=batch_per_gpu,shuffle=False,sampler=sampler_T,num_workers=1,drop_last=True)



        val_augmentor_rgb = get_augmentor(True, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=False)
        val_augmentor_flow = get_augmentor(True, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=True)

        train_dataset_s = VideoDataSet(self.datasets,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=True)
        trainloader_s = build_dataflow(train_dataset_s, is_train=True, batch_size=batch_per_gpu,is_distributed=True)
        train_dataset_t = VideoDataSet(self.unseen_dataset,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=True)
        trainloader_t = build_dataflow(train_dataset_t, is_train=True, batch_size=batch_per_gpu,is_distributed=True)

        val_augmentor_rgb = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=False)
        val_augmentor_flow = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=True)



        val_dataset_s = VideoDataSet(self.datasets,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        valoader_s = build_dataflow(val_dataset_s, is_train=False, batch_size=16,is_distributed=True)
        val_dataset_t = VideoDataSet(self.unseen_dataset,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        valoader_t = build_dataflow(val_dataset_t, is_train=False, batch_size=16,is_distributed=True)


        

        if self.modality=="flow":

            self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
                           temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update,
                           domain_mode=self.domain_mode,aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,
                           predict_synch=self.FLAGS.pred_synch, selfsupervised_lambda=self.FLAGS.self_lambda,
                           S_agent_flow=self.FLAGS.S_agent_flow, T_agent_flow=self.FLAGS.T_agent_flow,
                           S_agent_RGB=self.FLAGS.S_agent_RGB, T_agent_RGB=self.FLAGS.T_agent_RGB,
                           select_num=self.FLAGS.select_num, candidate_num=self.FLAGS.candidate_num,ts_flow=self.FLAGS.ts_flow,
                           tt_flow=self.FLAGS.tt_flow,ts_RGB=self.FLAGS.ts_RGB,tt_RGB=self.FLAGS.tt_RGB,batch_size=self.FLAGS.batch_size,epsilon_final=self.FLAGS.epsilon_final,
                           epsilon_start=self.FLAGS.epsilon_start, epsilon_decay=self.FLAGS.epsilon_decay, 
                           REPLAY_MEMORY=self.FLAGS.REPLAY_MEMORY, batch_dqn=self.FLAGS.batch_dqn,replace_target_iter =self.FLAGS.replace_target_iter)
            #print("check1")

        elif self.modality=="rgb":

            self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
                           temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update,
                           domain_mode=self.domain_mode,aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,
                           predict_synch=self.FLAGS.pred_synch, selfsupervised_lambda=self.FLAGS.self_lambda,
                           S_agent_flow=self.FLAGS.S_agent_flow, T_agent_flow=self.FLAGS.T_agent_flow,
                           S_agent_RGB=self.FLAGS.S_agent_RGB, T_agent_RGB=self.FLAGS.T_agent_RGB,
                           select_num=self.FLAGS.select_num, candidate_num=self.FLAGS.candidate_num,ts_flow=self.FLAGS.ts_flow,
                           tt_flow=self.FLAGS.tt_flow,ts_RGB=self.FLAGS.ts_RGB,tt_RGB=self.FLAGS.tt_RGB,batch_size=self.FLAGS.batch_size,epsilon_final=self.FLAGS.epsilon_final,
                           epsilon_start=self.FLAGS.epsilon_start, epsilon_decay=self.FLAGS.epsilon_decay, 
                           REPLAY_MEMORY=self.FLAGS.REPLAY_MEMORY, batch_dqn=self.FLAGS.batch_dqn,replace_target_iter =self.FLAGS.replace_target_iter)

        elif self.modality=="joint":

            self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
                           temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update,
                           domain_mode=self.domain_mode,aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,
                           predict_synch=self.FLAGS.pred_synch, selfsupervised_lambda=self.FLAGS.self_lambda,
                           S_agent_flow=self.FLAGS.S_agent_flow, T_agent_flow=self.FLAGS.T_agent_flow,
                           S_agent_RGB=self.FLAGS.S_agent_RGB, T_agent_RGB=self.FLAGS.T_agent_RGB,
                           select_num=self.FLAGS.select_num, candidate_num=self.FLAGS.candidate_num,ts_flow=self.FLAGS.ts_flow,
                           tt_flow=self.FLAGS.tt_flow,ts_RGB=self.FLAGS.ts_RGB,tt_RGB=self.FLAGS.tt_RGB,batch_size=self.FLAGS.batch_size,epsilon_final=self.FLAGS.epsilon_final,
                           epsilon_start=self.FLAGS.epsilon_start, epsilon_decay=self.FLAGS.epsilon_decay, 
                           REPLAY_MEMORY=self.FLAGS.REPLAY_MEMORY, batch_dqn=self.FLAGS.batch_dqn,replace_target_iter =self.FLAGS.replace_target_iter)

            #print("check1")

        #print("check2")

        #self.model.replace_logits(self.num_labels)
        #print("check3")

        #self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(), device_ids=[self.FLAGS.local_rank])

        if self.restoring:

            self.model.load_state_dict(torch.load(self.FLAGS.trained_path,map_location='cuda:{}'.format(local_rank)),strict=False)
        
        self.model.cuda()
        #print("check4")
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True,broadcast_buffers=True)


        if self.FLAGS.use_tfboard and dist.get_rank() == 0:
            from tensorboardX import SummaryWriter
            logname = self.results_dir+'/train'
            logger = SummaryWriter(logname)

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        #optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        #optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr, rho=0.9, eps=1e-6, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.5)

        #optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-8, weight_decay=0.0001, momentum=0.9, centered=False)
        
        #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
        count = 0
        steps = 1
        Done = False
        iters_per_epoch = len(trainloader_s)//self.FLAGS.batch_size
        print("iters_per_epoch",len(trainloader_s))

        input_rgb = torch.FloatTensor(1)
        input_flow = torch.FloatTensor(1)
        input_label = torch.LongTensor(1)
        input_synch = torch.FloatTensor(1)

        input_rgb = Variable(input_rgb)
        input_flow = Variable(input_flow)
        input_label = Variable(input_label)
        input_synch = Variable(input_synch)

        
        sync_loss = 0


        # val_data_T = Epic(self.num_labels, self.unseen_dataset,
        #                             temporal_window=self.FLAGS.temporal_window,
        #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                            synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)

        

        # valoader_t = DataLoader(val_data_T,batch_size=self.FLAGS.batch_size,shuffle=False,num_workers=0,drop_last=True)

        # val_data_S = Epic(self.num_labels, self.datasets,
        #                             temporal_window=self.FLAGS.temporal_window,
        #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                            synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)
        

        # valoader_s = DataLoader(val_data_S,batch_size=self.FLAGS.batch_size,shuffle=False,num_workers=0,drop_last=True)

        
        for epoch in range(self.FLAGS.epoch):

            self.model.train()
            loss_temp = 0
            start = time.time()
            # if epoch%7==0:
            #     #val_iter_s = iter(valoader_s)
            #     val_iter_t = iter(valoader_t)

            data_iter_s = iter(trainloader_s)
            data_iter_t = iter(trainloader_t)
            #with tqdm.tqdm(total=len(trainloader_s)) as t_bar:
            for step in range(len(trainloader_s)):
                try:
                    data_s = next(data_iter_s)

                except:
                    data_iter_s = iter(trainloader_s)
                    data_s = next(data_iter_s)
                try:
                    data_t = next(data_iter_t)
                except:
                    data_iter_t = iter(trainloader_t)
                    data_t = next(data_iter_t)

                if self.FLAGS.modality == "rgb":

                    p = float(steps) / self.FLAGS.max_step

                    lin = (2 / (1. + np.exp(-10. * p)) - 1) * self.FLAGS.lambda_in

                    image_s = input_rgb.resize_(data_s[0].size()).copy_(data_s[0])
                    input_flow.resize_(data_s[1].size()).copy_(data_s[1])
                    label_s = input_label.resize_(data_s[2].size()).copy_(data_s[2]).cuda()
                    input_synch.resize_(data_s[3].size()).copy_(data_s[3])
                    image_t = input_rgb.resize_(data_t[0].size()).copy_(data_t[0])
                    input_flow.resize_(data_t[1].size()).copy_(data_s[1])
                    label_t = input_label.resize_(data_t[2].size()).copy_(data_t[2]).cuda()
                    input_synch.resize_(data_t[3].size()).copy_(data_t[3])

                    # data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 
                    # images = torch.cat([image_s,image_t])
                    # images_rgb = torch.stack([data_augmentation.preprocess(x, True) for x in images]).permute(0,2,1,3,4).cuda()
                    #print("images_rgb",images_rgb.size())

                    #Refined_S, Refined_T, loss_S, loss_T, logits = train_step(self.model,self.FLAGS,lin,images_rgb,label_s,target_data=False)
                    #print("lRefined_Sogits",Refined_S)  
                    # log = image_s.size(0)

                    # logits = logits[:log,:]

                    if self.FLAGS.domain_mode == "DANN":


                        Refined, loss_S, loss_T, logits = self.model(inputs=images_rgb,eta=lin)
                        #print("refine",Refined.size())
                        num = Refined.size(0)//2
                        log = image_s.size(0)

                        logits = logits[:log,:]
                        logits = torch.squeeze(logits)
                        Refined_S = Refined[:num,:]
                        Refined_T = Refined[num:,:]

                        domain_loss_S = 0.5 * torch.mean(Refined_S ** 2)
                        #print("domain_loss_S",domain_loss_S)

                        domain_loss_T = 0.5 * torch.mean((1-Refined_T) ** 2)

                    #loc_loss = torch.nn.CrossEntropyLoss(logits, label_s)
                        cls_loss = F.binary_cross_entropy_with_logits(logits, label_s)
                        loss_dqn = (loss_S + loss_T)*0.2
                        domain_loss = domain_loss_S + domain_loss_T

                        loss = cls_loss + domain_loss + loss_dqn

                        
                    else:

                        _, logits = self.model(inputs=images_rgb)
                        #print("refine",Refined.size())
                        #num = Refined.size(0)//2
                        log = image_s.size(0)

                        logits = logits[:log,:]
                        logits = torch.squeeze(logits)

                        cls_loss = F.binary_cross_entropy_with_logits(logits, label_s)

                        loss = cls_loss

                        domain_loss = 0
                        loss_dqn = 0





    
                    # data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 
                    # images_rgb = torch.stack([data_augmentation.preprocess(x, True) for x in image]).permute(0,2,1,3,4).cuda()




                    # Refined_T,loss_T = train_step(self.model,self.FLAGS,lin,images_rgb,label,target_data=True)  
                
                elif self.FLAGS.modality == "flow":
                    input_rgb.resize_(data_s[0].size()).copy_(data_s[0])
                    image = input_flow.resize_(data_s[1].size()).copy_(data_s[1])
                    label = input_label.resize_(data_s[2].size()).copy_(data_s[2])
                    input_synch.resize_(data_s[3].size()).copy_(data_s[3])

                    # data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True)   
                    # images_flow = torch.stack([data_augmentation.preprocess(x, True) for x in image]).permute(0,2,1,3,4)  


                    # #Refined_S, loss_S, loc_loss = train_step(self.model,self.FLAGS,lin,images_flow,label,target_data=False)  


                    # image = input_rgb.resize_(data_t[0].size()).copy_(data_t[0])
                    # input_flow.resize_(data_t[1].size()).copy_(data_s[1])
                    # label = input_label.resize_(data_t[2].size()).copy_(data_t[2])
                    # input_synch.resize_(data_t[3].size()).copy_(data_t[3])
                    # data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 
                    # images_flow = torch.stack([data_augmentation.preprocess(x, True) for x in image]).permute(0,2,1,3,4)




                    Refined_T,loss_T = train_step(self.model,self.FLAGS,lin,images_flow,label,target_data=True)     

                elif self.FLAGS.modality == "joint":

                    p = float(steps) / self.FLAGS.max_step

                    lin = (2 / (1. + np.exp(-10. * p)) - 1) * self.FLAGS.lambda_in


                    #print("images_rgb",images_rgb.size())


                    if self.FLAGS.pred_synch:


                        image_s_rgb = input_rgb.resize_(data_s[0].size()).copy_(data_s[0])
                        image_s_flow = input_flow.resize_(data_s[1].size()).copy_(data_s[1])
                        label_s = input_label.resize_(data_s[2].size()).copy_(data_s[2])
                        
                        image_t_rgb = input_rgb.resize_(data_t[0].size()).copy_(data_t[0])
                        image_t_flow = input_flow.resize_(data_t[1].size()).copy_(data_s[1])
                        #label_t = input_label.resize_(data_t[2].size()).copy_(data_t[2]).cuda()




                        half = int(len(image_s_rgb)/2)
                        fixed_sample_rgb = image_s_rgb[:half]
                        fixed_sample_flow = image_s_flow[:half]
                        variate_sample_rgb = image_s_rgb[half:]
                        variate_sample_flow = image_s_flow[half:]
                        variate_sample_flow = torch.cat((variate_sample_flow[1:],variate_sample_flow[:1]),0)
                        image_s_flow = torch.cat((fixed_sample_flow,variate_sample_flow),0)


                        #########################shuffle##############################
                        shuffle_index=torch.randperm(len(image_s_rgb))
                        sychron_s =np.array([0]*half+[1]*half)   
                        sychron_s = sychron_s[shuffle_index]                    
                        sychron_s = torch.from_numpy(sychron_s).view(-1,1)
                        train_id = torch.nonzero(sychron_s.squeeze()==0).squeeze()
                        #print("sychron_s",sychron_s)
                        #sychron_s = torch.zeros(len(image_s_rgb),2).scatter_(1,sychron_s,1).float().cuda()
                        sychron_s = sychron_s.float().cuda()
                        
                        image_s_rgb = image_s_rgb[shuffle_index]
                        image_s_flow = image_s_flow[shuffle_index]
                        label_s = label_s[shuffle_index]
                        #print(label_s)
                        label_pred = label_s.cuda()
                    
                        label_s = F.one_hot(label_s,8).float().cuda()
                        #print(label_s)


                        #########################shuffle##############################


                        half = int(len(image_t_rgb)/2)
                        fixed_sample_rgb = image_t_rgb[:half]
                        fixed_sample_flow = image_t_flow[:half]
                        variate_sample_rgb = image_t_rgb[half:]
                        variate_sample_flow = image_t_flow[half:]
                        variate_sample_flow = torch.cat((variate_sample_flow[1:],variate_sample_flow[:1]),0)
                        image_t_flow = torch.cat((fixed_sample_flow,variate_sample_flow),0)


                    
                        sychron_t =np.array([0]*half+[1]*half)
                        sychron_t = sychron_t[shuffle_index]
                        sychron_t = torch.from_numpy(sychron_t).view(-1,1)
                        #sychron_t = torch.zeros(len(image_t_rgb),2).scatter_(1,sychron_t,1).cuda()
                        sychron_t = sychron_t.float().cuda()
                        shuffle_index=torch.randperm(len(image_t_rgb))
                        image_t_rgb = image_t_rgb[shuffle_index]
                        image_t_flow = image_t_flow[shuffle_index]
                        
                        

                        sychron = torch.cat((sychron_s,sychron_t),0)

                        # if dist.get_rank()==0:

                        #     print("sychron_s",sychron)

                        #########################shuffle##############################



                        '''
                        data_augmentation_rgb = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 
                        images_rgb = torch.cat([image_s_rgb,image_t_rgb])
                        images_rgb = torch.stack([data_augmentation_rgb.preprocess(x, True) for x in images_rgb]).permute(0,2,1,3,4).cuda()
                        images_flow = torch.cat([image_s_flow,image_t_flow])
                        data_augmentation_flow = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True) 
                        images_flow = torch.stack([data_augmentation_flow.preprocess(x, True) for x in images_flow]).permute(0,2,1,3,4).cuda()
                        '''

                        images_rgb = torch.cat([image_s_rgb,image_t_rgb]).cuda()
                        images_flow = torch.cat([image_s_flow,image_t_flow]).cuda()



                        #images_rgb = fixed_sample_rgb + variate_sample_rgb
                        # sychron = [1,0]*len(fixed_sample_rgb)+[0,1]*len(variate_sample_rgb)+[1,0]*len(fixed_sample_rgb)+[0,1]*len(variate_sample_rgb)
                        # sychron = np.array(sychron)
                        # sychron = torch.from_numpy(sychron).cuda()
                        # sychron =np.array([0]*half+[1]*half+[0]*half+[1]*half)
                        # sychron = torch.from_numpy(sychron).view(-1,1)
                        # sychron = torch.zeros(len(images_rgb),2).scatter_(1,sychron,1).cuda()
                        
                        #sychron = sychron.scatter_()

                    #print("images_flow",images_flow.size())
                    else:
                        image_s_rgb = input_rgb.resize_(data_s[0].size()).copy_(data_s[0])
                        image_s_flow = input_flow.resize_(data_s[1].size()).copy_(data_s[1])
                        label_s = input_label.resize_(data_s[2].size()).copy_(data_s[2])
                        # input_synch.resize_(data_s[3].size()).copy_(data_s[3])
                        image_t_rgb = input_rgb.resize_(data_t[0].size()).copy_(data_t[0])
                        image_t_flow = input_flow.resize_(data_t[1].size()).copy_(data_s[1])
                        # label_t = input_label.resize_(data_t[2].size()).copy_(data_t[2]).cuda()
                        # input_synch.resize_(data_t[3].size()).copy_(data_t[3])

                        # data_augmentation_rgb = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 
                        # images_rgb = torch.cat([image_s_rgb,image_t_rgb])
                        # images_rgb = torch.stack([data_augmentation_rgb.preprocess(x, True) for x in images_rgb]).permute(0,2,1,3,4).cuda()
                        # images_flow = torch.cat([image_s_flow,image_t_flow])
                        # data_augmentation_flow = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True) 
                        # images_flow = torch.stack([data_augmentation_flow.preprocess(x, True) for x in images_flow]).permute(0,2,1,3,4).cuda()


                        images_rgb = torch.cat([image_s_rgb,image_t_rgb]).cuda()
                        images_flow = torch.cat([image_s_flow,image_t_flow]).cuda()
                        label_pred = label_s.cuda()
                        label_s = F.one_hot(label_s,8).float().cuda()


                    if self.FLAGS.domain_mode == "DANN":


                        if self.FLAGS.pred_synch:

                            Refined_rgb, Refined_flow, loss_S_rgb, loss_T_rgb, loss_S_flow, \
                            loss_T_flow, logits_rgb, logits_flow,logits_synch = self.model(inputs=None,inputs_rgb=images_rgb,inputs_flow=images_flow,eta=lin)
                        else:


                            Refined_rgb, Refined_flow, loss_S_rgb, loss_T_rgb, loss_S_flow, \
                            loss_T_flow, logits_rgb, logits_flow = self.model(inputs=None,inputs_rgb=images_rgb,inputs_flow=images_flow,eta=lin)

                        #print("Refined_rgb",Refined_rgb)
                        # print("Refined_flow",Refined_flow.size())
    
                        num = Refined_flow.size(0)//2
                        log = image_s_flow.size(0)
                        logits_flow = logits_flow[:log,:]
                        logits_flow = torch.squeeze(logits_flow)
                        Refined_S_flow = Refined_flow[:num]
                        Refined_T_flow = Refined_flow[num:]
                        
                        # domain_loss_S_flow = 0.5 * torch.mean(Refined_S_flow ** 2)
                        # domain_loss_T_flow = 0.5 * torch.mean((1-Refined_T_flow) ** 2)
                        # domain_loss_flow = domain_loss_S_flow + domain_loss_T_flow

                        domain_label_flow = [0]*num+[1]*num
                        domain_label_flow = np.array(domain_label_flow)
                        
                        domain_label_flow = torch.from_numpy(domain_label_flow).float().cuda()
                        #domain_label_flow = F.one_hot(domain_label_flow).float().cuda()
                        #print("domain_label_flow",domain_label_flow)

                        num = Refined_rgb.size(0)//2
                        log = image_s_rgb.size(0)
                        logits_rgb = logits_rgb[:log,:]
                        logits_rgb = torch.squeeze(logits_rgb)
                        Refined_S_rgb = Refined_rgb[:num]
                        Refined_T_rgb = Refined_rgb[num:]

                        # domain_loss_S_rgb = 0.5 * torch.mean(Refined_S_rgb ** 2)
                        # domain_loss_T_rgb = 0.5 * torch.mean((1-Refined_T_rgb) ** 2)
                        # domain_loss_rgb = domain_loss_S_rgb + domain_loss_T_rgb


                        domain_label_rgb = [0]*num+[1]*num
                        domain_label_rgb = np.array(domain_label_rgb)
                        
                        domain_label_rgb = torch.from_numpy(domain_label_rgb).float().cuda()
                        #domain_label_rgb = F.one_hot(domain_label_rgb).float().cuda()
                        
                        #print("Refined_flow",Refined_flow.dtype)


                        loss_fn = torch.nn.BCELoss()
                        
                        #F.cross_entropy(Refined_flow,)
                        # domain_loss_flow = F.cross_entropy(Refined_flow, domain_label_flow)
                        # domain_loss_rgb = F.cross_entropy(Refined_rgb, domain_label_rgb)

                        domain_loss_flow = loss_fn(Refined_flow, domain_label_flow)
                        domain_loss_rgb = loss_fn(Refined_rgb, domain_label_rgb)

                    
                        # domain_loss_flow = F.binary_cross_entropy_with_logits(Refined_flow, domain_label_flow)
                        # domain_loss_rgb = F.binary_cross_entropy_with_logits(Refined_rgb, domain_label_rgb)
                        

                        # if True:
                        #     cse = torch.nn.CrossEntropyLoss()
                        #     logits = torch.mean(torch.stack((logits_rgb, logits_flow)), dim=0)
                        #     cls_loss = cse(logits, label_s)
                        # else:

                        #     cls_loss_flow = F.binary_cross_entropy_with_logits(logits_flow, label_s)
                        #     cls_loss_rgb = F.binary_cross_entropy_with_logits(logits_rgb, label_s)
                        #     cls_loss = cls_loss_flow + cls_loss_rgb

                        
                        loss_dqn = (loss_S_flow + loss_T_flow + loss_S_rgb + loss_T_rgb)
                        domain_loss = domain_loss_rgb + domain_loss_flow


                        if self.FLAGS.pred_synch:
                            

                            #sync_loss = torch.nn.CrossEntropyLoss(logits_synch, sychron)
                            logits = torch.mean(torch.stack((logits_rgb, logits_flow)), dim=0)       
                            cse = torch.nn.CrossEntropyLoss()
                            cls_loss =cse(logits[train_id,:], label_s[train_id,:])
                            criterion = torch.nn.BCEWithLogitsLoss()
                            sync_loss = criterion(logits_synch, sychron)
                            loss = cls_loss + sync_loss*self.FLAGS.self_lambda + domain_loss + loss_dqn                 


                            # sync_loss = F.binary_cross_entropy_with_logits(logits_synch, sychron)
                            # cls_loss = F.binary_cross_entropy_with_logits(logits[train_id,:], label_s[train_id,:])
                            # loss = cls_loss + domain_loss + loss_dqn + sync_loss

                            pred = torch.argmax(logits[train_id],1)
                            score = torch.eq(pred,label_pred[train_id]).long().float()
                            #print("score",score)
                            acc = torch.mean(score)
                            
                        else:

                            logits = torch.mean(torch.stack((logits_rgb, logits_flow)), dim=0)       
                            cse = torch.nn.CrossEntropyLoss()
                            cls_loss =cse(logits, label_s)


                            loss = cls_loss + domain_loss + loss_dqn
                        
                            pred = torch.argmax(logits,1)
                            score = torch.eq(pred,label_pred).long().float()
                            #print("score",score)
                            acc = torch.mean(score)

                        
                    else:
                        if self.FLAGS.pred_synch:

                            logits_rgb, logits_flow,logits_synch = self.model(inputs=None,inputs_rgb=images_rgb,inputs_flow=images_flow)
                        else:
                            logits_rgb, logits_flow = self.model(inputs=None,inputs_rgb=images_rgb,inputs_flow=images_flow)


                        log = image_s_flow.size(0)
                        logits_flow = logits_flow[:log,:]
                        logits_flow = torch.squeeze(logits_flow)
                                
                        log = image_s_rgb.size(0)
                        logits_rgb = logits_rgb[:log,:]
                        logits_rgb = torch.squeeze(logits_rgb)
                        domain_loss = 0
                        loss_dqn = 0
                        sync_loss = 0
                        

                        if True:
                            if self.FLAGS.pred_synch:
                                logits = torch.mean(torch.stack((logits_rgb, logits_flow)), dim=0)       
                                cse = torch.nn.CrossEntropyLoss()
                                cls_loss =cse(logits[train_id,:], label_s[train_id,:])
                                # sync_loss = cse(logits_synch, sychron)
                                criterion = torch.nn.BCEWithLogitsLoss()
                                sync_loss = criterion(logits_synch, sychron)
                                loss = cls_loss+sync_loss*self.FLAGS.self_lambda  
                                pred = torch.argmax(logits[train_id],1)
                                score = torch.eq(pred,label_pred[train_id]).long().float()
                                #print("score",score)
                                acc = torch.mean(score)                         
                            else:
                                cse = torch.nn.CrossEntropyLoss()
                                logits = torch.mean(torch.stack((logits_rgb, logits_flow)), dim=0)
                                # cls_loss = F.binary_cross_entropy_with_logits(logits, label_s)
                                cls_loss = cse(logits, label_s)
                                loss = cls_loss
                                pred = torch.argmax(logits,1)
                                score = torch.eq(pred,label_pred).long().float()
                                #print("score",score)
                                acc = torch.mean(score)       
                        
                        else:

                            cls_loss_flow = F.binary_cross_entropy_with_logits(logits_flow, label_s)
                            cls_loss_rgb = F.binary_cross_entropy_with_logits(logits_rgb, label_s)
                            cls_loss = cls_loss_flow + cls_loss_rgb

                        

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                eval=False
                ##store model
                if steps>7501 and steps % 50==1:
                    #lr_sched.step
                    if dist.get_rank() == 0:
                        torch.save(self.model.module.state_dict(), self.save_model+'/'+str(steps).zfill(6)+'.pt')
                ##write logging
                if steps<7501 and steps % 2000==1:
                    #lr_sched.step
                    if dist.get_rank() == 0:
                        torch.save(self.model.module.state_dict(), self.save_model+'/'+str(steps).zfill(6)+'.pt')
                ##write logging
                
                if steps % 2000==1 and steps<7501:
                    eval =True
                if steps>7501 and steps%50==1:
                    eval = True
    
                    #if dist.get_rank() == 0:
                if eval:
                    a = time.time()

                    val_iter_t = iter(valoader_t)
                    inputt = val_iter_t                   
                    val_iter_s = iter(valoader_s)
                    inputs = val_iter_s

                    if self.FLAGS.domain_mode == "DANN":
                        #valaccuracy, average_class,domainaccuracy = evaluate(self.model, self.FLAGS, inputs, lin, test=True, extra_info=True)
                        valaccuracy_t, average_class_t, domainaccuracy_t = evaluate(self.model, self.FLAGS, inputt, lin, test=True, extra_info=True)
                        if dist.is_initialized():
                            world_size = dist.get_world_size()
                            # dist.all_reduce(valaccuracy)
                            # dist.all_reduce(average_class)
                            # dist.all_reduce(domainaccuracy)
                            # valaccuracy /= world_size
                            # average_class /= world_size
                            # domainaccuracy /= world_size
                            dist.all_reduce(valaccuracy_t)
                            dist.all_reduce(average_class_t)
                            dist.all_reduce(domainaccuracy_t)
                            valaccuracy_t /= world_size
                            average_class_t /= world_size
                            domainaccuracy_t /= world_size

                    

                        if dist.get_rank() == 0:
                            #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f domainaccuracy: %f" % (datetime.datetime.now(), "Source", steps, valaccuracy, average_class,domainaccuracy))                       
                            print("(Val) %s: domain:%s step:%d accuracy:%f avg_class: %f domainaccuracy: %f" % (datetime.datetime.now(), "Target", steps, valaccuracy_t, average_class_t,domainaccuracy_t))
                    else:

                        #valaccuracy, average_class = evaluate(self.model, self.FLAGS, inputs, lin, test=True, extra_info=True)
                        valaccuracy_t, average_class_t = evaluate(self.model, self.FLAGS, inputt, lin, test=True, extra_info=True)
                        if dist.is_initialized():
                            world_size = dist.get_world_size()
                            # dist.all_reduce(valaccuracy)
                            # dist.all_reduce(average_class)      
                            # valaccuracy /= world_size
                            # average_class /= world_size                
                            dist.all_reduce(valaccuracy_t)
                            dist.all_reduce(average_class_t)
                            valaccuracy_t /= world_size
                            average_class_t /= world_size        
                        if dist.get_rank() == 0:                        
                            #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f" % (datetime.datetime.now(), "Source", steps, valaccuracy, average_class))
                            print("(Val) %s: domain:%s step:%d accuracy:%f avg_class: %f" % (datetime.datetime.now(), "Target", steps, valaccuracy_t, average_class_t))

                    
                    b = time.time()
                    if dist.get_rank() == 0:
                        print("tiem used eval",(b-a))
                        if self.FLAGS.use_tfboard:
                    
                            info = {
                                "valaccuracy_t":valaccuracy_t}




                            #.cpu().detach().numpy()
                            logger.add_scalars("logs_s_1/acc", info,steps) 

                    self.model.train() 
                        

                if dist.get_rank() == 0:
                    # if self.FLAGS.domain_mode=="DANN":
                    #     print('step: {:d} domain_loss: {:.4f} cls_loss: {:.4f} sync_loss: {:.4f} loss_dqn: {:.4f}  Tot Loss: {:.4f} Acc: {:.4f}'.format(steps,domain_loss, cls_loss, sync_loss, loss_dqn, loss,acc))
                    # else:

                    #     print('step: {:d} domain_loss: {:.4f} cls_loss: {:.4f} sync_loss: {:.4f} loss_dqn: {:.4f}  Tot Loss: {:.4f} Acc: {:.4f}'.format(steps,domain_loss, cls_loss, sync_loss, loss_dqn, loss,acc))

                    if self.FLAGS.use_tfboard:
                        if self.FLAGS.pred_synch:
                            info = {
                                'loss': loss,
                                'domain_loss': domain_loss,
                                'cls_loss': cls_loss,
                                'sync_loss':sync_loss
                            }
                        elif self.FLAGS.domain_mode=="DANN":
                            info = {
                            'loss': loss,
                            'domain_loss': domain_loss,
                            'cls_loss': cls_loss,
                            'loss_dqn': loss_dqn,
                        }
                        else:
                            info = {
                                'loss': loss,
                                'cls_loss': cls_loss
                            }
                        accu = {'acc':acc}

                        #.cpu().detach().numpy()
                        logger.add_scalars("logs_s_1/losses", info,steps)
                        logger.add_scalars("logs_s_1/accuracy", accu,steps)                         

                
                # t_bar.update(1)
                if steps==self.FLAGS.max_step:
                    Done = True
                    break
                steps+=1   
                

            end = time.time()
            
            
            if dist.get_rank() == 0:
                print("Epoch (%d) Done! time used for Epoch:%.4f" % (epoch,(end-start)))
            scheduler.step()
            if Done==True:
                break

        if self.FLAGS.use_tfboard and dist.get_rank() == 0:
            logger.close()

    def test(self):
        start = time.time()

        dist.init_process_group(backend='nccl')
        #torch.cuda.set_device(args.local_rank)


        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        #torch.set_num_threads(4)


        # val_data_T = Epic(self.num_labels, self.unseen_dataset,
        #                             temporal_window=self.FLAGS.temporal_window,
        #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                            synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)

        

        # valoader_t = DataLoader(val_data_T,batch_size=self.FLAGS.batch_size,shuffle=False,num_workers=0,drop_last=True)

        # val_data_S = Epic(self.num_labels, self.datasets,
        #                             temporal_window=self.FLAGS.temporal_window,
        #                             rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
        #                            synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)
        

        # valoader_s = DataLoader(val_data_S,batch_size=self.FLAGS.batch_size,shuffle=False,num_workers=0,drop_last=True)

        val_augmentor_rgb = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=False)
        val_augmentor_flow = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=True)



        # val_dataset_s = VideoDataSet(self.datasets,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        # valoader_s = build_dataflow(val_dataset_s, is_train=False, batch_size=16,is_distributed=True)
        val_dataset_t = VideoDataSet(self.unseen_dataset,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        valoader_t = build_dataflow(val_dataset_t, is_train=False, batch_size=16,is_distributed=True)



        self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
                                temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update,
                                domain_mode=self.domain_mode,aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,
                                predict_synch=self.FLAGS.pred_synch, selfsupervised_lambda=self.FLAGS.self_lambda,
                                S_agent_flow=self.FLAGS.S_agent_flow, T_agent_flow=self.FLAGS.T_agent_flow,
                                S_agent_RGB=self.FLAGS.S_agent_RGB, T_agent_RGB=self.FLAGS.T_agent_RGB,
                                select_num=self.FLAGS.select_num, candidate_num=self.FLAGS.candidate_num,ts_flow=self.FLAGS.ts_flow,
                                tt_flow=self.FLAGS.tt_flow,ts_RGB=self.FLAGS.ts_RGB,tt_RGB=self.FLAGS.tt_RGB,batch_size=self.FLAGS.batch_size,epsilon_final=self.FLAGS.epsilon_final,
                                epsilon_start=self.FLAGS.epsilon_start, epsilon_decay=self.FLAGS.epsilon_decay, 
                                REPLAY_MEMORY=self.FLAGS.REPLAY_MEMORY, batch_dqn=self.FLAGS.batch_dqn,replace_target_iter =self.FLAGS.replace_target_iter)

        #self.model.load_state_dict(torch.load(self.FLAGS.trained_path),strict=False)
        self.model.load_state_dict(torch.load(self.FLAGS.trained_path,map_location='cuda:{}'.format(local_rank)),strict=False)

        #self.model.load_state_dict(torch.load(self.FLAGS.trained_path,map_location='cuda:{}'.format(local_rank)),strict=False)
        self.model.cuda()
        #self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True,broadcast_buffers=True)



        # val_iter_s = iter(valoader_s)
        # inputs = val_iter_s

        # valaccuracy, average_class = evaluate(self.model, self.FLAGS, inputs, 0.5, test=True, extra_info=True)
        # print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f" % (datetime.datetime.now(), "Source", self.FLAGS.step, valaccuracy, average_class))

        val_iter_t = iter(valoader_t)
        inputt = val_iter_t  

        prep = time.time()
        print("time used for pre:",(prep-start))

       
        if self.FLAGS.domain_mode =="DANN":
            #valaccuracy, average_class, domainaccuracy = evaluate(self.model, self.FLAGS, inputs, 0.5, test=True, extra_info=True)
            valaccuracy_t, average_class_t, domainaccuracy_t = evaluate(self.model, self.FLAGS, inputt, 0.5, test=True, extra_info=True)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                # dist.all_reduce(valaccuracy)
                # dist.all_reduce(average_class)
                # dist.all_reduce(domainaccuracy)
                # valaccuracy /= world_size
                # average_class /= world_size
                # domainaccuracy /= world_size
                dist.all_reduce(valaccuracy_t)
                dist.all_reduce(average_class_t)
                dist.all_reduce(domainaccuracy_t)
                valaccuracy_t /= world_size
                average_class_t /= world_size
                domainaccuracy_t /= world_size
            if dist.get_rank()==1:
                #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class: %f domainaccuracy: %f" % (datetime.datetime.now(), "Source", self.FLAGS.step, valaccuracy, average_class,domainaccuracy)) 
                
                print("(Val) %s: domain:%s step:%d accuracy:%f avg_class: %f domainaccuracy: %f" % (datetime.datetime.now(), "Target", self.FLAGS.step, valaccuracy_t, average_class_t,domainaccuracy_t)) 

        else:
            #valaccuracy, average_class = evaluate(self.model, self.FLAGS, inputs, 0.5, test=True, extra_info=True)
            valaccuracy_t, average_class_t = evaluate(self.model, self.FLAGS, inputt, 0.5, test=True, extra_info=True)
            
            if dist.is_initialized():
                world_size = dist.get_world_size()
                # dist.all_reduce(valaccuracy)
                # dist.all_reduce(average_class)     
                # valaccuracy /= world_size
                # average_class /= world_size
                dist.all_reduce(valaccuracy_t)
                dist.all_reduce(average_class_t)
                valaccuracy_t /= world_size
                average_class_t /= world_size


            if dist.get_rank()==1:
                #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f" % (datetime.datetime.now(), "Source", self.FLAGS.step, valaccuracy, average_class))

                
                print("(Val) %s: domain:%s step:%d accuracy:%f avg_class: %f" % (datetime.datetime.now(), "Target", self.FLAGS.step, valaccuracy_t, average_class_t)) 

    
                        
        #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f" % (datetime.datetime.now(), "Source", steps, valaccuracy, average_class))
        
        # valaccuracy, average_class = evaluate(self.model, self.FLAGS, inputt, 0.5, test=True, extra_info=True)
        #print("(Val) %s: domain:%s step:%d accuracy:%f avg_class %f" % (datetime.datetime.now(), "Target", self.FLAGS.step, valaccuracy, average_class))
        end = time.time()
        print("time used:",(end-start))

if __name__ == '__main__':
    # need to add argparse

    args, train_dir, results_dir = parse_args()



    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    train_test = TrainTestScript(args, results_dir, train_dir)
    if args.train:
        print("train",args.train)
        train_test.train()
    else:
        train_test.test()



