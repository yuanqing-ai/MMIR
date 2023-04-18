import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from data_gen.preprocessing import DataAugmentation
import time





def train_step(model,flags,lin,inputs,labels,target_data):

    # if target_data:

    #     Refined, loss_T, _ = model(inputs,agent="target")
    #     #domain_loss = 0.5 * torch.mean((1-Refined) ** 2)

    # else:
    #     Refined, loss_S, logits = model(inputs,agent="source")
    #     #domain_loss = 0.5 * torch.mean(Refined ** 2)
    #     logits = torch.squeeze(logits)
    #     print("logits",logits.size())
    #     loc_loss = F.binary_cross_entropy_with_logits(logits, labels)

    # if target_data:

    #     return Refined, loss_T
    # else:
    #     return Refined, loss_S, loc_loss

    Refined, loss_S, loss_T, logits = model(inputs=inputs)
    #print("refine",Refined.size())
    num = Refined.size(0)//2
    logits = torch.squeeze(logits)
    Refined_s = Refined[:num,:]
    Refined_t = Refined[num:,:]

    return Refined_s, Refined_t, loss_S, loss_T, logits


def evaluate(model, flags, inputsd, lin, test=True, extra_info=False):
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
    np_domain_logits_all_rgb = []
    np_domain_logits_all_flow = []
    
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
                        np_logits_rgb, np_logits_flow, d_score_total_rgb, d_score_total_flow = model(inputs=None,inputs_rgb=input_rgb,inputs_flow=input_flow)
                        np_domain_logits_all_rgb.append(d_score_total_rgb)
                        np_domain_logits_all_flow.append(d_score_total_flow)
                    else:
                        
                        np_logits_rgb, np_logits_flow = model(inputs=None,inputs_rgb=input_rgb,inputs_flow=input_flow)
                # print("d_score_total_rgb",d_score_total_rgb)
                # print("d_score_total_flow",d_score_total_flow)
                # print("np_logits_rgb",np_logits_rgb)
                # print("np_logits_flow",np_logits_flow)
                
                logits = torch.stack((np_logits_rgb,np_logits_flow))
                #print("logits",logits.shape)
                logits = torch.sum(logits,dim=0)
                #print("logits",logits)

                np_logits = F.softmax(logits,dim=1)
                #print("np_logits",np_logits)

                    
                np_logits_all.append(np_logits.detach().cpu().numpy())
                b = time.time()
                #print("time ra",(b-a))
                


            end = time.time()
            #print("time used one",(end-start))
            # average predictions and compute top1 accuracy for each sample
            #print("np_logits_all",np_logits_all)
            np_test_one_hot = np.zeros((np_test_d.shape[0], 2), dtype=np.int32)
            np_test_one_hot[np.arange(np_test_d.shape[0]), np_test_d] = 1

            
            correct_np, predicted_np = correct(np_logits_all, np_test_y)
            #print("correct_np",correct_np)
            

            # average domain prediction and compute top1 accuracy
            # np_test_one_hot = np.zeros((np_test_d.shape[0], 2), dtype=np.int32)
            # np_test_one_hot[np.arange(np_test_d.shape[0]), np_test_d] = 1
            if flags.domain_mode=="DANN":

                correct_domain_np, predicted_domain_np = correct(np_domain_logits_all, np_test_one_hot)
                

                correct_domain_list = np.concatenate((correct_domain_list, correct_domain_np))
            correct_list = np.concatenate((correct_list, correct_np))

            label_list = np.concatenate((label_list, label_np))
        




    # Macro average accuracies
    valaccuracy = correct_list.mean()
    


    #Compute per class recall
    perclass = pd.DataFrame({'correct': correct_list, 'label': label_list}).groupby('label')['correct'].mean()

    if extra_info:
        if flags.domain_mode=="DANN":
            domainaccuracy = correct_domain_list.astype(float).mean()

            return torch.tensor(valaccuracy).cuda(), torch.tensor(perclass.mean()).cuda() , torch.tensor(domainaccuracy).cuda()
        else:
            return torch.tensor(valaccuracy).cuda(), torch.tensor(perclass.mean()).cuda()

    else:
        if flags.domain_mode=="DANN":
            domainaccuracy = correct_domain_list.astype(float).mean()
            return torch.tensor(valaccuracy).cuda(), torch.tensor(perclass.mean()).cuda() , torch.tensor(domainaccuracy).cuda()
        else:

            return torch.tensor(valaccuracy).cuda(), torch.tensor(perclass.mean()).cuda() #, domainaccuracy

    
        
