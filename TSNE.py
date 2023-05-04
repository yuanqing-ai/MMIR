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

def tsne(model, flags, inputsd, lin, test=True, extra_info=False):
    """Evaluate action performance

    Returns top1 action accurcy, top1 domain accuracy, Average class recall. Extra information including filenames,
    labels, feature can be returned by setting extra_info=True.

    (sess) tensorflow session
    (model) model to evalutae
    (flags) option specifed in tensorflow flags
    (batch_gen) data generator to evaluate
    (lin) Gradient Reversal layer hyper-parameter
    (test) Use training or test dataset. default test
    (out_features) output feature represenation produced before classification.
    (extra_info) return extra information, including feature representation, filename, labels, etc
    """

    # Average predictions from multiple timesteps (axis=0)
    # Compute top1 accuracy
    def correct(prediction, label_onehot):
        if np.all(np.equal(prediction,-1)):
            return np.zeros(label_onehot.shape[0],dtype=np.float32), np.zeros(label_onehot.shape[0],dtype=np.float32)
        #print("prediction",prediction)
        prediction_avg = np.mean(prediction, 1)
        #print("prediction_avg",prediction_avg)
        predicted = np.argmax(prediction_avg, 1)
        #print("predicted",predicted)
        label = np.argmax(label_onehot, 1)
        #print("label",label)
        return np.equal(label, predicted), predicted

    # Repeat examples to fit required batch size,
    # returns padded examples and original length of batch
    def pad_batch(x_rgb, x_flow, y):
        num_real_examples = x_rgb.shape[0]
        tt = flags.batch_size//flags.num_gpus
        while x_rgb.shape[0] < tt:
            x_rgb = np.concatenate([x_rgb, x_rgb], axis=0)
            x_flow = np.concatenate([x_flow, x_flow], axis=0)
            y = np.concatenate([y, y], axis=0)
        x_rgb = x_rgb[:tt]
        x_flow = x_flow[:tt]
        y = y[:tt]
        # d = d[:(self.flags.batch_size)]
        return x_rgb, x_flow, y, num_real_examples

    #done = False
    correct_list = []
    predicted_list = []
    feature_list = []
    filenames_list = []
    label_list = []
    correct_domain_list = []
    model.eval()
    np_features_all_rgb = []
    np_features_all_flow = []
    
    # Get batch



    # Evaluate on a number of timesteps per action segment

    if flags.modality=="rgb":

    
        data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False) 

        if extra_info:
            itera = len(inputsd)
        else:
            itera = 4
        for i in range(itera):
            #flags.batch_size//flags.num_gpus
            star = time.time()


            np_logits_all = []
            np_domain_logits_all = []
            np_features = []
            np_filenames = None
            inputa = next(inputsd)
            np_test_x_rgb_all = inputa[0]
            np_test_x_flow_all = inputa[1] 
            np_test_y = inputa[2]
            np_test_y = np_test_y.squeeze()
            np_test_d = np.array([1] * np_test_x_rgb_all.shape[0])
            label_np = np.argmax(np_test_y, axis=1)
            
            
            np_test_x_rgb_all = torch.stack([data_augmentation.preprocess(x, False) for x in np_test_x_rgb_all]).permute(0,2,1,3,4)

            for np_test_x_rgb in np_test_x_rgb_all:
                a = time.time()
                
                synch = [True]*np_test_x_rgb.shape[0]


                if flags.modality == "rgb":
                    #labels = torch.from_numpy(np_test_y)
                    #np_test_x_rgb = np_test_x_rgb_all[i]
                    np_test_x_rgb = np_test_x_rgb.unsqueeze(0)
                    #print("np_test_x_rgb_",np_test_x_rgb.size())
                    np_test_x_rgb=np_test_x_rgb.chunk(5, dim=2)
                    np_test_x_rgb = torch.cat([i for i in np_test_x_rgb])
                    #label_np = np.argmax(np_test_y, axis=1)


                    # print("np_test_x_rgb",np_test_x_rgb.size())
                    
                    
                    inputs = np_test_x_rgb.cuda()      
                    #labels = Variable(labels.cuda())  
                    
                elif flags.modality == "flow":
                    images_flow = np_test_x_flow
                    labels = np_test_y
                    dataset_ident = np_test_d
                    dropout = 1.0
                    synch = synch
                    flip_weight = lin 
                    data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True)   
                    images_flow = torch.stack([data_augmentation.preprocess(x, False) for x in images_flow]).permute(0,2,1,3,4)   
                    # inputs = Variable(images_flow.cuda())               
                    # labels = Variable(torch.tensor(labels).cuda())     
                    inputs = images_flow.cuda()            
                    labels = torch.tensor(labels).cuda()           

                elif flags.modality == "joint":
                    images_rgb = np_test_x_rgb
                    images_flow = np_test_x_flow
                    labels = np_test_y
                    dataset_ident = np_test_d
                    dropout = 1.0
                    synch = synch
                    flip_weight = lin 
                    data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False)   
                    images_flow = torch.stack([data_augmentation.preprocess(x, False) for x in images_flow]).permute(0,2,1,3,4) 
                    data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True) 
                    images_rgb = torch.stack([data_augmentation.preprocess(x, False) for x in images_rgb]).permute(0,2,1,3,4)     
                    inputs = Variable(images_rgb.cuda())
                    inputs = Variable(images_flow.cuda())
                else:
                    raise Exception("Unknown modality"+flags.modality)

                # Run Validation
                
                with torch.no_grad():
                    np_domain_logits, np_logits = model(inputs=inputs)


                np_logits = F.softmax(np_logits,dim=1)

                    
                np_logits_all.append(np_logits.detach().cpu().numpy())
                b = time.time()
                print("time used ra",(b-a))
            en = time.time()
            print("time used it",(en-star))


            # average predictions and compute top1 accuracy for each sample
            correct_np, predicted_np = correct(np_logits_all, np_test_y)
            #print("correct_np",correct_np.shape)

            # average domain prediction and compute top1 accuracy
            np_test_one_hot = np.zeros((np_test_d.shape[0], 2), dtype=np.int32)
            np_test_one_hot[np.arange(np_test_d.shape[0]), np_test_d] = 1

            correct_list = np.concatenate((correct_list, correct_np))

            label_list = np.concatenate((label_list, label_np))
        
    if flags.modality=="joint":



        np_test_x_rgb_all = torch.FloatTensor(1)
        np_test_x_flow_all = torch.FloatTensor(1)
        np_test_y = torch.FloatTensor(1)
        #input_synch = torch.FloatTensor(1)

        np_test_x_rgb_all = Variable(np_test_x_rgb_all)
        np_test_x_flow_all = Variable(np_test_x_flow_all)
        np_test_y = Variable(np_test_y)
        #input_synch = Variable(input_synch)


        if extra_info:
            itera = len(inputsd)
            #print(itera)
        else:
            itera = 2
        for i in range(itera):
            #flags.batch_size//flags.num_gpus
            start = time.time()


            np_logits_all = []

            np_domain_logits_all = []
            np_features = []
            np_filenames = None
            inputa = next(inputsd)
            # np_test_x_rgb_all = inputa[0]
            # np_test_x_flow_all = inputa[1] 
            np_test_y = inputa[2]
            np_test_y = F.one_hot(np_test_y,8)
            #np_test_y = np_test_y.squeeze()
            np_test_d = np.array([1] * np_test_x_rgb_all.shape[0])
            label_np = np.argmax(np_test_y, axis=1)
           # print(np_test_y)

            np_test_x_rgb_all = np_test_x_rgb_all.resize_(inputa[0].size()).copy_(inputa[0])
            np_test_x_flow_all = np_test_x_flow_all.resize_(inputa[1].size()).copy_(inputa[1])
            #np_test_y = np_test_y.cuda()
            np_test_d = np.array([1] * np_test_x_rgb_all.shape[0])


            
            #np_test_x_rgb_all = torch.stack([data_augmentation.preprocess(x, False) for x in np_test_x_rgb_all]).permute(0,2,1,3,4)
            # np_test_x_flow_all = torch.stack([data_augmentation_flow.preprocess(x, False) for x in np_test_x_flow_all]).permute(0,2,1,3,4)
            # np_test_x_rgb_all = torch.stack([data_augmentation_rgb.preprocess(x, False) for x in np_test_x_rgb_all]).permute(0,2,1,3,4)  
            #print("np_test_x_rgb_all",np_test_x_rgb_all.size()) 
        
            #print("time aug",(aug1-aug))     

            for np_test_x_rgb,np_test_x_flow in zip(np_test_x_rgb_all,np_test_x_flow_all):
                a = time.time()
                
                synch = [True]*np_test_x_rgb.shape[0]
                np_test_x_rgb = np_test_x_rgb.unsqueeze(0)
                np_test_x_rgb=np_test_x_rgb.chunk(5, dim=2)
                np_test_x_rgb = torch.cat([i for i in np_test_x_rgb])
                np_test_x_flow = np_test_x_flow.unsqueeze(0)
                np_test_x_flow = np_test_x_flow.chunk(5, dim=2)
                np_test_x_flow = torch.cat([i for i in np_test_x_flow]) 

                                    
                # input_rgb = Variable(np_test_x_rgb.cuda())
                # input_flow = Variable(np_test_x_flow.cuda())
                input_rgb = np_test_x_rgb.cuda()
                input_flow = np_test_x_flow.cuda()
                # Run Validation
                #model.eval()
                with torch.no_grad():
                    if flags.domain_mode=="DANN":
                        np_logits_rgb, np_logits_flow, d_score_total_rgb, d_score_total_flow, features_rgb, features_flow = model(inputs=None,inputs_rgb=input_rgb,inputs_flow=input_flow)
                    else:
                        
                        np_logits_rgb, np_logits_flow, features_rgb, features_flow = model(inputs=None,inputs_rgb=input_rgb,inputs_flow=input_flow)
                # print("d_score_total_rgb",d_score_total_rgb)
                # print("d_score_total_flow",d_score_total_flow)
                # print("np_logits_rgb",np_logits_rgb)
                # print("np_logits_flow",np_logits_flow)
                np_features_all_rgb.append(features_rgb)
                np_features_all_flow.append(features_flow)
    np.save('tsne_rgb',np_features_all_rgb)
    np.save('tsne_flow',np_features_all_flow)

class T_sne():

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

    def t-SNE():
        val_augmentor_rgb = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=False)
        val_augmentor_flow = get_augmentor(False, 224, disable_scaleup=True,threed_data=True,
                                        is_flow=True)
        val_dataset_s = VideoDataSet(self.datasets,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        valoader_s = build_dataflow(val_dataset_s, is_train=False, batch_size=16,is_distributed=False)
        val_dataset_t = VideoDataSet(self.unseen_dataset,self.FLAGS.rgb_data_path,self.FLAGS.flow_data_path,modality='rgb',transform_rgb=val_augmentor_rgb,transform_flow=val_augmentor_flow, is_train=False)
        valoader_t = build_dataflow(val_dataset_t, is_train=False, batch_size=16,is_distributed=False)
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

        self.model.load_state_dict(torch.load(self.FLAGS.trained_path,map_location='cuda:{}'.format(local_rank)),strict=False)

        self.model.cuda()
        val_iter_t = iter(valoader_t)
        inputt = val_iter_t  
        tsne(self.model, self.FLAGS, inputt, 0.5, test=True, extra_info=True)
        val_iter_s = iter(valoader_s)
        inputs = val_iter_s  
        tsne(self.model, self.FLAGS, inputs, 0.5, test=True, extra_info=True)
if __name__ == '__main__':
    # need to add argparse

    args, train_dir, results_dir = parse_args()



    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    tsne = T_sne(args, results_dir, train_dir)
    tsne.t-SNE()