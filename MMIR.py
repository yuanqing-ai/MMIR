import torch
import torch.nn as nn
import math
from pytorch_i3d import InceptionI3d, Domain_Classifier,Sync
from net_utils import grad_reverse,revgrad
import numpy as np

import DQN

class Model(nn.Module):


    '''

    (num_gpus) number of gpus to split the batch. batch size must be a multiple of num_gpus.
     (num_labels) Output side of classification head.
     (feature_layer) Layer of i3d for feature extraction/MMD/Domain discrimination. Deafult after avg pool 'features'
     (temporal window) Number of frames in video segment
     (batch_norm_update) Adjusts how quickly batch norm statistics are updated
     (modality) Modalities in model. Either "rgb", "flow" or late fusion "joint"
     (domain_mode) Mode for domain adaptation and/or allowing unlabelled target data during pretraining,
     (learning_rate) learning rate
     (steps_per_update) Number of forward-backward passes before SGD step
     (gradient_reversal) Negate gradient in backward pass after domain discriminators,
     (aux_classifier) Use multiple classification heads, overrided as true if using MCD baseline
     (synchronised) synchronise flow and rgb augmentations
     (predict_synch) Use correspondence classification head
     (selfsupervised_lambda) weighting of self-supervised alignment loss,
     (lambda_class) Override weighting of classification loss. Default 0.2 unless pretraining lambda_class=1.0.

     Train network: A series of properties on model object can be used to train the network:
     model - dictionary of model outputs/logits
     losses - dictionary of losses
     placeholders - dictinoary of input places to the model
     predictions - softmax predictions of the model and metrics
     zero_grads - zero accumulated gradients
     accum_grads - forward-backward pass and accumulated gradients
     train_op - SGD step with accumulated gradients

     Save/load models: A series of methods on model object to save/load models
     init_savers - initialise model savers
     restore_model_train - restore model for training. Options "pretrain" (only base model, no classification heads),
                           "model" (only base model with action classification head) or "continue" (all weights in model).
     restore_model_test - restore model for testing.

     get_summaries - return TensorFlow summaries

     '''

    def __init__(self, num_gpus, S_agent_flow, T_agent_flow,S_agent_RGB, T_agent_RGB,select_num, 
                 candidate_num,ts_flow,tt_flow,ts_RGB,tt_RGB,batch_size, 
                 epsilon_final, epsilon_start, epsilon_decay, REPLAY_MEMORY, num_labels=10,  feature_layer='features',
                 temporal_window=16, batch_norm_update=0.9, modality="joint", domain_mode="None",
                 steps_per_update=1, gradient_reversal=True,
                 aux_classifier=False, synchronised=False, predict_synch=False,  selfsupervised_lambda=5,
                 lambda_class=False, batch_dqn = 200, replace_target_iter = 6 ):

        super(Model, self).__init__()

        self.selfsupervised_lambda=selfsupervised_lambda
        self.synchronised = synchronised
        self.modality = modality
        self.num_gpus = num_gpus
        self.steps_per_update = steps_per_update
        self.num_labels = num_labels
        self.network_output = self.num_labels
        self.temporal_window = temporal_window
        self.batch_norm_update = batch_norm_update
        self.pred_synch = predict_synch
        self.batch_size = batch_size
        self.__placeholders = None
        self.__predictions = None
        self.__losses = None
        self.__model = None
        self.__train_op = None
        self.__savers_rgb = None
        self.__savers_flow = None
        self.__savers_joint = None
        self.feat_level = feature_layer
        self.summaries = []
        self.replace_target_iter = replace_target_iter
        self.aux_classifier = aux_classifier

        ##parameters for RL
        self.epsilon_final = epsilon_final
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.S_agent_flow = S_agent_flow
        self.T_agent_flow = T_agent_flow
        self.S_agent_RGB = S_agent_RGB
        self.T_agent_RGB = T_agent_RGB
        self.ts_flow = ts_flow
        self.ts_RGB = ts_RGB
        self.tt_flow = tt_flow
        self.tt_RGB = tt_RGB
        self.candidate_num = candidate_num
        self.epsilon_by_epoch = lambda epoch_idx: self.epsilon_final + (self.epsilon_start - \
            self.epsilon_final) * math.exp(-1. * epoch_idx / self.epsilon_decay)
        self.iter_dqn = 0

        self.epsilon_by_epoch_T = lambda epoch_idx: self.epsilon_final + (self.epsilon_start - \
            self.epsilon_final) * math.exp(-1. * epoch_idx / self.epsilon_decay)
        self.action_num = candidate_num
        self.iter_dqn_T_RGB = 0   
        self.iter_dqn_S_RGB = 0  
        self.iter_dqn_T_flow = 0   
        self.iter_dqn_S_flow = 0 

        self.select_num = select_num
        self.state_size = 1024*self.candidate_num
        self.pred_size = 1024*2
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.batch_dqn = batch_dqn

        if not (domain_mode == "DANN"  or domain_mode == "BN" or
                domain_mode == "MMD" or domain_mode == "MCD" or domain_mode == "None" or domain_mode == "Pretrain" or domain_mode == "PretrainM"):
            raise Exception("Invalid domain_mode option: {}".format(domain_mode))

        # Modality specification
        if self.modality == "joint":
            self.joint_classifier = True
            self.rgb = True
            self.flow = True
        elif self.modality == "rgb" or self.modality == "flow":
            self.joint_classifier = False
            self.rgb = self.modality == "rgb"
            self.flow = self.modality == "flow"
        else:
            raise Exception("Invalid modality option: {}".format(self.modality))

        self.flip_weight = gradient_reversal

        if domain_mode == "None" or domain_mode == "Pretrain":
            self.target_data = False
        else:
            self.target_data = True

        self.domain_loss = (domain_mode == "DANN")
        self.mmd = (domain_mode == "MMD")
        self.MaxClassDiscrepany = (domain_mode == "MCD")
        self.bn_align = (domain_mode == "BN")

        if lambda_class:
            self.softmax_lambda = 1.0
        else:
            self.softmax_lambda = 1.0 if (domain_mode == "Pretrain" or domain_mode == "PretrainM") else 0.2

        if self.domain_loss:
            if self.modality == "joint":


                self.Domain_Classifier_rgb = Domain_Classifier(1024,context=False)
                self.Domain_Classifier_flow = Domain_Classifier(1024,context=False)
            else:
                self.Domain_Classifier = Domain_Classifier(context=False)


        if self.domain_loss:

            if self.modality == 'flow':

                self.cs = DQN.DQN(self.action_num,self.state_size)
                self.t_s = DQN.DQN(self.action_num,self.state_size)
                self.rs = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)
                self.ct = DQN.DQN(self.action_num,self.state_size)
                self.t_t = DQN.DQN(self.action_num,self.state_size)
                self.rt = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)             
                self.count_S = self.iter_dqn_T_flow
                self.count_T = self.iter_dqn_S_flow
                self.ts = self.ts_flow
                self.tt = self.tt_flow
                


            elif self.modality == 'rgb':

                self.cs = DQN.DQN(self.action_num,self.state_size)
                self.t_s = DQN.DQN(self.action_num,self.state_size)
                self.rs = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)
                self.t_t = DQN.DQN(self.action_num,self.state_size)
                self.ct = DQN.DQN(self.action_num,self.state_size)
                self.rt = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size) 
                self.count_S = self.iter_dqn_T_RGB
                self.count_T = self.iter_dqn_S_RGB
                self.ts = self.ts_RGB
                self.tt = self.tt_RGB
                print("print(model)",self.cs)

            else:
                self.cs_rgb = DQN.DQN(self.action_num,self.state_size)
                self.t_s_rgb = DQN.DQN(self.action_num,self.state_size)
                self.rs_rgb = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)
                self.t_t_rgb = DQN.DQN(self.action_num,self.state_size)
                self.ct_rgb = DQN.DQN(self.action_num,self.state_size)
                self.rt_rgb = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size) 
                self.count_S_rgb = self.iter_dqn_T_RGB
                self.count_T_rgb = self.iter_dqn_S_RGB
                self.ts_rgb = self.ts_RGB
                self.tt_rgb = self.tt_RGB

                self.cs_flow = DQN.DQN(self.action_num,self.state_size)
                self.t_s_flow = DQN.DQN(self.action_num,self.state_size)
                self.rs_flow = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)
                self.ct_flow = DQN.DQN(self.action_num,self.state_size)
                self.t_t_flow = DQN.DQN(self.action_num,self.state_size)
                self.rt_flow = DQN.ReplayBuffer(self.REPLAY_MEMORY,self.state_size)             
                self.count_S_flow = self.iter_dqn_T_flow
                self.count_T_flow = self.iter_dqn_S_flow
                self.ts_flow = self.ts_flow
                self.tt_flow = self.tt_flow


        if self.modality=="flow":

            self.model = InceptionI3d(400, in_channels=2)

            self.model.load_state_dict(torch.load('models/flow_imagenet.pt'),strict=False)
            self.model.replace_logits(self.num_labels)
            print("check1")

        elif self.modality=="rgb":

            self.model = InceptionI3d(400, in_channels=3)

            self.model.load_state_dict(torch.load('models/rgb_imagenet.pt'),strict=False)
            self.model.replace_logits(self.num_labels)

        elif self.modality=="joint":
            self.model_rgb = InceptionI3d(400, in_channels=3)

            self.model_rgb.load_state_dict(torch.load('models/rgb_imagenet.pt'),strict=False)
            self.model_rgb.replace_logits(self.num_labels)

            self.model_flow = InceptionI3d(400, in_channels=2)

            self.model_flow.load_state_dict(torch.load('models/flow_imagenet.pt'),strict=False)
            self.model_flow.replace_logits(self.num_labels)
        
        if self.pred_synch:
            self.sync_cls = Sync(self.pred_size)

    def forward(self,inputs,inputs_rgb=None,inputs_flow=None,target=False,eta=1):


        lossQ = 0.0
        loss_S = 0.0
        loss_T = 0.0
        loss_S_rgb = 0.0
        loss_T_rgb = 0.0
        loss_S_flow = 0.0
        loss_T_flow = 0.0
        Refined_S = []
        Refined_T = []
        eta = torch.tensor(eta)
        

        if self.aux_classifier:
            if self.modality=="joint": 
                # inputs_rgb = inputs[:(inputs.size(0)//2),:]
                # inputs_flow = inputs[(inputs.size(0)//2):,:]

                logits_flow, features_flow, aux_logits_flow = self.model_flow(inputs_flow)
                logits_rgb, features_rgb, aux_logits_rgb = self.model_rgb(inputs_rgb)
                logits_flow = logits_flow.squeeze()
                logits_rgb = logits_rgb.squeeze()
                if self.pred_synch:
                    

                    logits_synch = self.sync_cls(torch.cat((features_rgb, features_flow), dim=1))
            else:

                logits, features, aux_logits = self.model(inputs)
                logits = logits.squeeze()
        else:
            
            if self.modality=="joint":
                # inputs_rgb = inputs[:(inputs.size(0)//2),:]
                # inputs_flow = inputs[(inputs.size(0)//2):,:]

                logits_rgb, features_rgb = self.model_rgb(inputs_rgb)

                logits_flow, features_flow  = self.model_flow(inputs_flow)
                logits_flow = logits_flow.squeeze()
                logits_rgb = logits_rgb.squeeze()
                if self.pred_synch:
                    #print("features_rgb",features_rgb.size())

                    logits_synch = self.sync_cls(torch.cat((features_rgb, features_flow), dim=1))

               
            else:
                logits, features = self.model(inputs)
                logits = logits.squeeze()


        #print("feartures",feartures.size())

        
        #pooled_feat = tf.reshape(feartures, shape=(tf.shape(d_feat_RGB)[0],-1))
        #pooled_feat = feartures.view(feartures.size(0), -1)
        # pooled_feat_D = features.squeeze(2)
        # #pooled_feat = feartures.squeeze()
        # pooled_feat = pooled_feat_D.view(pooled_feat_D.size(0),-1)
        
        # if self.context:
        #     d_instance, _ = self.Domain_Classifier(grad_reverse(pooled_feat, lambd=eta))
        #     #if target:
        #         #d_instance, _ = self.netD_pixel(grad_reverse(pooled_feat, lambd=eta))
        #         #return d_pixel#, diff
        #     d_score_total,feat = self.Domain_Classifier(pooled_feat.detach())
        # else:
        '''
        d_score_total = self.Domain_Classifier(pooled_feat_D.detach())
        d_instance = self.Domain_Classifier(grad_reverse(pooled_feat_D, lambd=eta))
        '''
        # pooled_feat_D_rev = revgrad(pooled_feat_D,eta)
        # d_score_total = self.Domain_Classifier(pooled_feat_D.detach())
        # d_instance = self.Domain_Classifier(pooled_feat_D_rev)
        # d_instance = d_instance.squeeze()
        # Refined = d_instance
        # logits = logits.squeeze()


        if self.domain_loss:
            if self.modality=="joint":

                pooled_feat_D_rgb = features_rgb.squeeze(2)
                pooled_feat_rgb = pooled_feat_D_rgb.view(pooled_feat_D_rgb.size(0),-1)

                
                pooled_feat_D_rev_rgb = revgrad(pooled_feat_D_rgb,eta)
                d_score_total_rgb = self.Domain_Classifier_rgb(pooled_feat_D_rgb.detach())
                d_instance_rgb = self.Domain_Classifier_rgb(pooled_feat_D_rev_rgb)
                d_instance_rgb = d_instance_rgb.squeeze()
                Refined_rgb = d_instance_rgb
                #print("d_instance_rgb",d_instance_rgb)

                # num_samples_rgb =d_instance_rgb.size(0)//2
                # pooled_feat_S_rgb = pooled_feat_rgb[:num_samples_rgb, :]
                # pooled_feat_T_rgb = pooled_feat_rgb[num_samples_rgb:, :]
                # #print("pooled_feat_S",pooled_feat_S)
                # d_instance_S_rgb = d_instance_rgb[:num_samples_rgb, :]
                # d_instance_T_rgb = d_instance_rgb[num_samples_rgb:, :]
                # d_score_total_S_rgb = d_score_total_rgb[:num_samples_rgb, :]
                # d_score_total_T_rgb = d_score_total_rgb[num_samples_rgb:, :]

                num_samples_rgb =d_instance_rgb.size(0)//2
                pooled_feat_S_rgb = pooled_feat_rgb[:num_samples_rgb]
                pooled_feat_T_rgb = pooled_feat_rgb[num_samples_rgb:]
                #print("pooled_feat_S",pooled_feat_S)
                d_instance_S_rgb = d_instance_rgb[:num_samples_rgb]
                d_instance_T_rgb = d_instance_rgb[num_samples_rgb:]
                d_score_total_S_rgb = d_score_total_rgb[:num_samples_rgb]
                d_score_total_T_rgb = d_score_total_rgb[num_samples_rgb:]


                if self.training and self.S_agent_RGB:

                    d_instance_q = d_instance_S_rgb

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    d_score_total_q = d_score_total_S_rgb
                    #print("d_score_total_q",d_score_total_q.size())
                    '''
                    for img in range(num_samples_rgb):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    '''

                    #print("d_score_total_qs",d_score_total_qs)
                    d_score_total_qs = np.array(d_score_total_q.cpu().detach().numpy())
                    pooled_feat_s = pooled_feat_S_rgb           
                    #print("------------------begin selecting in the source-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
        

                            epsilon = self.epsilon_by_epoch(self.count_S_rgb)
                            action_index = self.cs_rgb.act(state,epsilon,select_list)
                            # state = state.flatten()
                            # state = state.reshape((1,self.state_size))
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.ts_rgb:
                                reward = -1
                            else:
                                reward = 1     
                            #print("reward:",reward)
                            next_state = state
                            #next_state = torch.tensor(state)  
                            #print("next_state:",next_state.size())          
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rs_rgb.push(state,action_index,reward,next_state,done,select_list)
                            self.count_S_rgb = self.count_S_rgb+1
                            state = next_state
                            #print("s_rgb")
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rs_rgb)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.cs_rgb,self.t_s_rgb,self.rs_rgb,self.batch_dqn)         
                    if np.mod(self.count_S_rgb,self.replace_target_iter)==0:
                        DQN.update_target(self.cs_rgb,self.t_s_rgb)

                    d_instance_refine = d_instance_q[select_index]

                    idx_rgb = select_index

                    Refined_S_rgb = d_instance_refine
                    #print("Refined_S",Refined_S)
                    loss_S_rgb = lossQ
                

                if self.training and self.T_agent_RGB:
                    d_instance_q = d_instance_T_rgb
                    d_score_total_q = d_score_total_T_rgb

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    '''
                    for img in range(num_samples_rgb):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    '''
                    pooled_feat_s = pooled_feat_T_rgb  
                    d_score_total_qs = np.array(d_score_total_q.cpu().detach().numpy())      
                    #print("------------------begin selecting in the target-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
                    
                        
                            epsilon = self.epsilon_by_epoch_T(self.count_T_rgb)
                            action_index = self.ct_rgb.act(state,epsilon,select_list)

                
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.tt_rgb:
                                reward = 1
                            else:
                                reward = -1     
                            #print("reward:",reward)
                            #next_state = torch.tensor(state)
                            next_state = state            
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rt_rgb.push(state,action_index,reward,next_state,done,select_list)
                            self.count_T_rgb = self.count_T_rgb+1
                            state = next_state
                            #print("t_rgb")
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rt_rgb)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.ct_rgb,self.t_t_rgb,self.rt_rgb,self.batch_dqn)         
                    if np.mod(self.count_T_rgb,self.replace_target_iter)==0:
                        DQN.update_target(self.ct_rgb,self.t_t_rgb)

                    d_instance_refine = d_instance_q[select_index]

                    Refined_T_rgb = d_instance_refine
                    #print("Refined_T",Refined_T)
                    loss_T_rgb = lossQ


                pooled_feat_D_flow = features_flow.squeeze(2)
                pooled_feat_flow = pooled_feat_D_flow.view(pooled_feat_D_flow.size(0),-1)

                
                pooled_feat_D_rev_flow = revgrad(pooled_feat_D_flow,eta)
                d_score_total_flow = self.Domain_Classifier_flow(pooled_feat_D_flow.detach())
                d_instance_flow = self.Domain_Classifier_flow(pooled_feat_D_rev_flow)
                d_instance_flow = d_instance_flow.squeeze()
                Refined_flow = d_instance_flow

                # num_samples_flow =d_instance_flow.size(0)//2
                # pooled_feat_S_flow = pooled_feat_flow[:num_samples_flow, :]
                # pooled_feat_T_flow = pooled_feat_flow[num_samples_flow:, :]
                # #print("pooled_feat_S",pooled_feat_S)
                # d_instance_S_flow = d_instance_flow[:num_samples_flow, :]
                # d_instance_T_flow = d_instance_flow[num_samples_flow:, :]
                # d_score_total_S_flow = d_score_total_flow[:num_samples_flow, :]
                # d_score_total_T_flow = d_score_total_flow[num_samples_flow:, :]

                num_samples_flow =d_instance_flow.size(0)//2
                pooled_feat_S_flow = pooled_feat_flow[:num_samples_flow]
                pooled_feat_T_flow = pooled_feat_flow[num_samples_flow:]
                #print("pooled_feat_S",pooled_feat_S)
                d_instance_S_flow = d_instance_flow[:num_samples_flow]
                d_instance_T_flow = d_instance_flow[num_samples_flow:]
                d_score_total_S_flow = d_score_total_flow[:num_samples_flow]
                d_score_total_T_flow = d_score_total_flow[num_samples_flow:]
                


                if self.training and self.S_agent_RGB:


                    d_instance_q = d_instance_S_flow

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    d_score_total_q = d_score_total_S_flow
                    #print("d_score_total_q",d_score_total_q.size())
                    '''
                    for img in range(num_samples_flow):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    '''
                    #print("d_score_total_qs",d_score_total_qs)
                    d_score_total_qs = np.array(d_score_total_q.cpu().detach().numpy())
                    pooled_feat_s = pooled_feat_S_flow           
                    #print("------------------begin selecting in the source-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
        

                            epsilon = self.epsilon_by_epoch(self.count_S_flow)
                            action_index = self.cs_flow.act(state,epsilon,select_list)
                            # state = state.flatten()
                            # state = state.reshape((1,self.state_size))
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.ts_flow:
                                reward = -1
                            else:
                                reward = 1     
                            #print("reward:",reward)
                            next_state = state
                            #next_state = torch.tensor(state)  
                            #print("next_state:",next_state.size())          
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rs_flow.push(state,action_index,reward,next_state,done,select_list)
                            self.count_S_flow = self.count_S_flow+1
                            state = next_state
                            #print("s_flow")
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rs_flow)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.cs_flow,self.t_s_flow,self.rs_flow,self.batch_dqn)         
                    if np.mod(self.count_S_flow,self.replace_target_iter)==0:
                        DQN.update_target(self.cs_flow,self.t_s_flow)

                    d_instance_refine = d_instance_q[select_index]

                    idx_flow = select_index

                    Refined_S_flow = d_instance_refine
                    #print("Refined_S",Refined_S)
                    loss_S_flow = lossQ
                

                if self.training and self.T_agent_RGB:
                    d_instance_q = d_instance_T_flow
                    d_score_total_q = d_score_total_T_flow

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    '''
                    for img in range(num_samples_flow):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    '''
                    pooled_feat_s = pooled_feat_T_flow  
                    d_score_total_qs = np.array(d_score_total_q.cpu().detach().numpy())         
                    #print("------------------begin selecting in the target-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
                    
                        
                            epsilon = self.epsilon_by_epoch_T(self.count_T_flow)
                            action_index = self.ct_flow.act(state,epsilon,select_list)

                
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.tt_flow:
                                reward = 1
                            else:
                                reward = -1     
                            #print("reward:",reward)
                            #next_state = torch.tensor(state)
                            next_state = state            
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rt_flow.push(state,action_index,reward,next_state,done,select_list)
                            self.count_T_flow = self.count_T_flow+1
                            state = next_state
                            #print("t_flow")
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rt_flow)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.ct_flow,self.t_t_flow,self.rt_flow,self.batch_dqn)         
                    if np.mod(self.count_T_flow,self.replace_target_iter)==0:
                        DQN.update_target(self.ct_flow,self.t_t_flow)

                    d_instance_refine = d_instance_q[select_index]

                    Refined_T_flow = d_instance_refine
                    #print("Refined_T",Refined_T)
                    loss_T_flow = lossQ



                





            

            else:

                pooled_feat_D = features.squeeze(2)
                pooled_feat = pooled_feat_D.view(pooled_feat_D.size(0),-1)

                
                pooled_feat_D_rev = revgrad(pooled_feat_D,eta)
                d_score_total = self.Domain_Classifier(pooled_feat_D.detach())
                d_instance = self.Domain_Classifier(pooled_feat_D_rev)
                d_instance = d_instance.squeeze()
                Refined = d_instance

                num_samples =d_instance.size(0)//2
                pooled_feat_S = pooled_feat[:num_samples, :]
                pooled_feat_T = pooled_feat[num_samples:, :]
                #print("pooled_feat_S",pooled_feat_S)
                d_instance_S = d_instance[:num_samples, :]
                d_instance_T = d_instance[num_samples:, :]
                d_score_total_S = d_score_total[:num_samples, :]
                d_score_total_T = d_score_total[num_samples:, :]

                #print("self.training",self.training)

                if self.training and self.S_agent_RGB:

                    d_instance_q = d_instance_S

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    d_score_total_q = d_score_total_S
                    #print("d_score_total_q",d_score_total_q.size())
                    for img in range(num_samples):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    #print("d_score_total_qs",d_score_total_qs)
                    d_score_total_qs = np.array(d_score_total_qs)
                    pooled_feat_s = pooled_feat_S           
                    #print("------------------begin selecting in the source-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
        

                            epsilon = self.epsilon_by_epoch(self.count_S)
                            action_index = self.cs.act(state,epsilon,select_list)
                            # state = state.flatten()
                            # state = state.reshape((1,self.state_size))
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.ts:
                                reward = -1
                            else:
                                reward = 1     
                            #print("reward:",reward)
                            next_state = state
                            #next_state = torch.tensor(state)  
                            #print("next_state:",next_state.size())          
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rs.push(state,action_index,reward,next_state,done,select_list)
                            self.count_S = self.count_S+1
                            state = next_state
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rs)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.cs,self.t_s,self.rs,self.batch_dqn)         
                    if np.mod(self.count_S,self.replace_target_iter)==0:
                        DQN.update_target(self.cs,self.t_s)

                    d_instance_refine = d_instance_q[select_index]
                    idx_s = select_index

                    Refined_S = d_instance_refine
                    #print("Refined_S",Refined_S)
                    loss_S = lossQ
                

                if self.training and self.T_agent_RGB:
                    d_instance_q = d_instance_T
                    d_score_total_q = d_score_total_T

                    #d_score_total_q = d_score_total.split(128,0)
                    d_score_total_qs = []
                    for img in range(num_samples):
                        temp = torch.mean(d_score_total_q[img],dim=2)
                        #temp1 = torch.mean(temp,dim=1)
                        d_score_total_qs.append(torch.mean(temp,dim=1).cpu().detach().numpy())
                    pooled_feat_s = pooled_feat_T  
                    d_score_total_qs = np.array(d_score_total_qs)         
                    #print("------------------begin selecting in the target-----------------------")
                    select_iter = int(pooled_feat_s.shape[0]/self.candidate_num)
                    total_index = list(range(0,pooled_feat_s.shape[0]))
                    np.random.shuffle(total_index)
                    select_index = []
                    for eposide in range(select_iter):
                        #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                        select_list = list(range(0,self.candidate_num))
                        batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                        state = pooled_feat_s[batch_idx]
                        #print("state.shape:",state.shape)
                        d_score = d_score_total_qs[batch_idx]       
                        #print("d_score.shape:",d_score.shape)                
                        for it in range(self.select_num):
                            #print("#########begain the %d-th selection################" % (it))      
                    
                        
                            epsilon = self.epsilon_by_epoch_T(self.count_T)
                            action_index = self.ct.act(state,epsilon,select_list)

                
                            #print("action_index:",action_index)
                            #action_episode.append(action_index)
                            try:
                                select_list.remove(action_index)
                            except:
                                print("select_list:",select_list)
                                print("action_index:",action_index)
                                print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                continue
                            #print("the %d-th select, action_index is %d"%(it,action_index))
                            if d_score[action_index] > self.tt:
                                reward = 1
                            else:
                                reward = -1     
                            #print("reward:",reward)
                            #next_state = torch.tensor(state)
                            next_state = state            
                            next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                            if it==(self.select_num-1):
                                done = 1
                            else:
                                done = 0
                            self.rt.push(state,action_index,reward,next_state,done,select_list)
                            self.count_T = self.count_T+1
                            state = next_state
                        select_index = select_index + [batch_idx[i] for i in select_list]
                    if len(self.rt)>self.batch_dqn:
                        lossQ = DQN.compute_td_loss(self.ct,self.t_t,self.rt,self.batch_dqn)         
                    if np.mod(self.count_T,self.replace_target_iter)==0:
                        DQN.update_target(self.ct,self.t_t)

                    d_instance_refine = d_instance_q[select_index]
                    idx_t = select_index
                    Refined_T = d_instance_refine
                    #print("Refined_T",Refined_T)
                    loss_T = lossQ

            if self.domain_loss and self.training:
                if self.modality=="joint" and self.S_agent_RGB:

                    Refined_rgb = torch.cat((Refined_S_rgb,Refined_T_rgb), dim=0)
                    Refined_flow = torch.cat((Refined_S_flow,Refined_T_flow), dim=0)


                else:

                    if self.S_agent_RGB and self.T_agent_RGB:
                        Refined = torch.cat((Refined_S,Refined_T), dim=0)
        
                    elif self.S_agent_RGB:
                        Refined = torch.cat([Refined_S,d_instance_T], dim=0)
                        #Refined = torch.stack(Refined_S,d_instance_T)
                        loss_T = 1
                        print('loss_T:',loss_T)
                        print('loss_S:',loss_S)
                        
                    elif self.T_agent_RGB:
                        Refined = torch.cat([d_instance_T,Refined_T], dim=0)
                        
                        #Refined = torch.stack(d_instance_S,Refined_T)
                        loss_S = 1
                        print('loss_T:',loss_T)
                        print('loss_S:',loss_S)
                
                    else:
                        #Refined = d_instance
                        loss_S = 0.0
                        loss_T = 0.0
        #print("Refined",Refined.dtype)
        # print("loss_S",loss_S.dtype)

        # print("loss_T",loss_T.dtype)


        #print("logits",logits.dtype)

        # if agent=="source":
        #     return Refined, loss_S, logits

        # elif agent=="target":

        #     return Refined, loss_T, logits
        # else:

        #     return logits

        if self.domain_loss and self.training:
            if self.modality=="joint":
                if self.pred_synch:
                    return Refined_rgb, Refined_flow, loss_S_rgb, loss_T_rgb, loss_S_flow, loss_T_flow, logits_rgb, logits_flow, logits_synch
                else:
                    return Refined_rgb, Refined_flow, loss_S_rgb, loss_T_rgb, loss_S_flow, loss_T_flow, logits_rgb, logits_flow
                    #return logits_rgb, logits_flow
            else:

                return Refined, loss_S, loss_T, logits
        else:
            if self.modality=="joint":
                if self.pred_synch and self.training:
                    return logits_rgb, logits_flow, logits_synch
                elif self.domain_loss:
                    return logits_rgb, logits_flow, d_score_total_rgb, d_score_total_flow
                else:
                    return logits_rgb, logits_flow
                
            else:

                return Refined, logits






