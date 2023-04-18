import numpy as np
from random import randint
import random
import pandas as pd
from random import shuffle
from itertools import groupby
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2


class OversamplingWrapper(torch.utils.data.Dataset):
    def __init__(self, folder_dataset, oversampling_size=500):
        self.folder_dataset = folder_dataset
        self.oversampling_size = oversampling_size
        self.num_classes = 8

        self.class_idx_to_sample_ids = {i: [] for i in range(self.num_classes)}
        for idx, (_, class_id) in enumerate(folder_dataset.samples):
            self.class_idx_to_sample_ids[class_id].append(idx)

    def __len__(self):
        return self.num_classes * self.oversampling_size

    def __getitem__(self, index):
        class_id = index % self.num_classes
        sample_idx = random.sample(self.class_idx_to_sample_ids[class_id], 1)
        return self.folder_dataset[sample_idx[0]]


def default_loader(path,modality,test):
    #print("path",path.shape)
    path = np.squeeze(path)
    #segment = torch.empty()

    if test==False:
        #print(path)
        if modality=="RGB":
            segment = torch.cat([torch.unsqueeze(torch.FloatTensor(cv2.imread(f).astype(np.float32)),0) for f in path],dim=0)
        else:

            path_u = path[:, 0]
            path_v = path[:, 1]
            segment_u = torch.cat([torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(cv2.imread(f,0).astype(np.float32)),0),-1) for f in path_u],dim=0)
            segment_v = torch.cat([torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(cv2.imread(f,0).astype(np.float32)),0),-1) for f in path_v],dim=0)
            segment = torch.cat([segment_u, segment_v], dim=-1)

        #segment = torch.cat([torch.unsqueeze(torch.FloatTensor(np.array(Image.open(f).convert(modality))), 0) for f in path], dim=0)
        segment = segment.float()
        return segment
    else:

        if modality=="RGB":
            segment =torch.cat([torch.cat([torch.unsqueeze(torch.FloatTensor(np.array(Image.open(f))), 0) for f in pat],dim=0) for pat in path])
        else:

            path_u = path[:,:, 0]
            path_v = path[:,:, 1]
            #segment =torch.cat([torch.cat([torch.unsqueeze(torch.FloatTensor(np.array(Image.open(f))), 0) for f in pat],dim=0) for pat in path])
            segment_u = torch.cat([torch.cat([torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(cv2.imread(f,0).astype(np.float32)),0),-1) for f in path],dim=0) for path in path_u])
            segment_v = torch.cat([torch.cat([torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(cv2.imread(f,0).astype(np.float32)),0),-1) for f in path],dim=0) for path in path_v])
            #segment_v = torch.cat([torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(cv2.imread(f,0).astype(np.float32)),0),-1) for f in path_v],dim=0)
            segment = torch.cat([segment_u, segment_v], dim=-1)

   
    
        segment = segment.float()
        #print(segment)
        return segment



class Epic(Dataset):

    def __init__(self,num_labels,filename, temporal_window=16,
                flow_data_path="flow_frames_parent/flow",
                rgb_data_path="rgb_frames_parent/frames",
                synchronised=False, random_sync=False,test=False, loader= default_loader,modality='RGB'):

        self.test = test
        self.random_sync=random_sync
        self.synchronised = synchronised
        self.temporal_window = temporal_window
        self.flow_data_path = flow_data_path+"/"
        self.rgb_data_path = rgb_data_path+"/"
        self.filename = filename
        self.num_labels = num_labels

        dataset_data, dataset_labels = self._parse_inputs_df(
            filename+"_train.pkl")
        dataset_data_test, dataset_labels_test = self._parse_inputs_df(
            filename+"_test.pkl")

        dataset_data_total = np.arange(dataset_data.shape[0])
        dataset_test_total = np.arange(dataset_data_test.shape[0])

        dataset_data_train = (dataset_data,dataset_labels)
        dataset_data_test = (dataset_data_test, dataset_labels_test)
        self.dataset_data = {False: dataset_data_train, True:dataset_data_test}
        dataset_data_train_total = dataset_data_total
        dataset_data_test_total = dataset_test_total
        self.dataset_total = {False: dataset_data_train_total,True: dataset_data_test_total}


        self.dataset_data, self.dataset_labels = self.dataset_data[self.test]
        self.dataset_total = self.dataset_total[self.test]
        self.loader = loader
        #print("self.dataset_labels",self.dataset_labels)

    def __getitem__(self, index):

        #print("index",index)
        if self.test==False:

            batch_rgb, batch_flow, batch_labels, sychron = self.nextBatch(index)
            return self.loader(batch_rgb,'RGB',self.test),self.loader(batch_flow,'L',self.test),batch_labels,sychron
        else:

            _,batch_rgb,batch_flow,batch_labels = self.nextBatchEval(index)
            #print("self.loader(batch_rgb,'RGB',self.test)",self.loader(batch_rgb,'RGB',self.test).size())
            # print("batch_rgb,",batch_rgb)
            # print("batch_flow",batch_flow)

            return self.loader(batch_rgb,'RGB',self.test),self.loader(batch_flow,'L',self.test),batch_labels


        
        

    def __len__(self):

        #print("len(self.dataset_total)",self.dataset_total.shape)
        return len(self.dataset_total)

    def _parse_inputs_df(self, filename):
            """ Read annotation file """
            df = pd.read_pickle(filename)
            data = []
            for _, line in df.iterrows():
                image = [line['participant_id']+"/"+line['video_id'], line['start_frame'], line['stop_frame']]
                labels = line['verb_class']
                one_hot = np.zeros(self.num_labels)
                one_hot[labels] = 1.0
                #print(image)
                data.append((image[0],image[1],image[2], one_hot))

            segment,start,end, softmaxlabels = list(zip(*data))

            # get labels for all
            labels = list(softmaxlabels)
            train = list(zip(segment,start,end))
            train = np.array(train)
            labels = np.array(labels)
            #print(labels)
            return train, labels
    
    def nextBatch(self, sample_idx_t):
        """ Get next training samples
            loops through datasets, randomly sampling data without replacement to add to batch."""
        #batch_size = len(sample_idx_t)
        dataset_data = self.dataset_data
        dataset_labels = self.dataset_labels
        dataset_total = self.dataset_total
        #file = self.filename
        #Done = 0

        # Find path to dataset
        #if len(dataset_total) > batch_size:
        #sample_idx_t = np.random.choice(range(dataset_total.shape[0]),size = batch_size,replace=False)
        sample_idx = dataset_total[sample_idx_t]
            #self.dataset_total[test] = np.delete(dataset_total, sample_idx_t, axis=0)
        # else:
        #     sample_idx = dataset_total
        #     remaining = batch_size - sample_idx.shape[0]
        #     dataset_total_temp = np.arange(dataset_data.shape[0])
        #     dataset_total_temp = np.delete(dataset_total_temp, sample_idx, axis=0)
        #     sample_idx_t = np.random.choice(range(dataset_total_temp.shape[0]), size=remaining, replace=False)
        #     sample_idx = np.concatenate((sample_idx,dataset_total_temp[sample_idx_t]))
        #     self.dataset_total[test] = np.delete(dataset_total_temp, sample_idx_t, axis=0)
        #     Done = 1
        #     print("Done Epoch")

        sample = dataset_data[sample_idx]
        #print(sample)

        if self.synchronised:
            sychron = [True]*len(sample)
        else:
            sychron = [False]*len(sample)
        #sample = [self.sample_segment(filen, synchronise=to_sync) for (filen, to_sync) in zip(sample, sychron)]
        sample = [self.sample_segment(sample, synchronise=sychron)] 
        sample_rgb, sample_flow = zip(*sample)
        #print(sample_rgb)
        #print(sample_flow)
        if self.random_sync:
            half = int(len(sample)/2)
            fixed_sample_rgb = sample_rgb[:half]
            fixed_sample_flow = sample_flow[:half]
            variate_sample_rgb = sample_rgb[half:]
            variate_sample_flow = sample_flow[half:]
            variate_sample_flow = variate_sample_flow[1:] + variate_sample_flow[:1]
            sample_flow = fixed_sample_flow + variate_sample_flow
            sample_rgb = fixed_sample_rgb + variate_sample_rgb
            sychron = [True]*len(fixed_sample_rgb)+[False]*len(variate_sample_rgb)
        elif self.synchronised:
            sychron = [True] * len(sample)
        else:
            sychron = [True] * len(sample)
        sample_labels = dataset_labels[sample_idx]

        batch_labels = sample_labels
        #print("batch_labels",batch_labels)
        batch_rgb = list(sample_rgb)
        batch_flow = list(sample_flow)

        sychron = np.array(sychron)

        # Shuffle Batch
        # combined = list(zip(batch_rgb,batch_flow,batch_labels,sychron))
        # shuffle(combined)
        # batch_rgb, batch_flow, batch_labels,sychron = list(zip(*combined))

        #batch_rgb = np.array(batch_rgb)
        # batch_flow = np.array(batch_flow)
        #print("batch_labels",batch_rgb.shape)
        


        return batch_rgb,batch_flow,batch_labels,sychron

    def sample_segment(self, s, synchronise=False):
        """ Samples rgb and flow frame windows from a video segment s.
            Sampling temporal windows randomly in video segment.
            s = ["filename", start_frame, end_frame]"""
        #print(s[1],s[2])
        def _path_to_dataset(flow):
            if flow:
                left = self.flow_data_path
            else:
                left = self.rgb_data_path
            right = "/frame_"
            numframe = 10
            return left, right, numframe

        #Optical flow stacking
        #returns u,v frames for EPIC Kitchens
        def flow_filename(frameno,num_stack=1):
            left, right,fill_frame = _path_to_dataset(True)
            left_frame = frameno - int((num_stack-1)/2)
            right_frame = frameno + int(num_stack/2)
            filename = []
            for no in range(left_frame, right_frame + 1):
                filename.append(left + str(s[0]) + "/u" + right + str(no).zfill(fill_frame) + ".jpg")
                filename.append(left + str(s[0]) + "/v" + right + str(no).zfill(fill_frame) + ".jpg")
            return filename

        #return RGB frame for EPIC Kitchens
        def rgb_filename(frameno):
            left, right, fill_frame = _path_to_dataset(False)
            filename = left + str(s[0]) + right + str(frameno).zfill(fill_frame) + ".jpg"
            return filename

        def c3d_sampling():
            num_sample_frame = self.temporal_window
            half_sample_frame = int(self.temporal_window/2)
            segment_images = []
            segment_flow = []
            step = 2
            #print(s)
            segment_start = int(s[1]) + (step*half_sample_frame)
            segment_end = int(s[2])+1 - (step*half_sample_frame)
            #print("segment_start",segment_start)
            
            #print("segment_end",segment_end)

            #if unable to keep all frames in sample inside segment, allow sampling outside of segment
            if segment_start >= segment_end:
                segment_start = int(s[1])#
                segment_end = int(s[2])
            # make sure sampling is not bellow frame 1
            if segment_start <= half_sample_frame*step+1:
                segment_start = half_sample_frame*step+2

            if synchronise:
                center_frame_rgb = center_frame_flow = randint(segment_start,segment_end)
            else:
                center_frame_rgb = randint(segment_start, segment_end)
                center_frame_flow = randint(segment_start, segment_end)

            for no in range(center_frame_rgb - (step*half_sample_frame),center_frame_rgb + (step*half_sample_frame),step):
                segment_images.append(rgb_filename(no))
            for no in range(center_frame_flow - (step*half_sample_frame),center_frame_flow + (step*half_sample_frame),step):
                segment_flow.append(flow_filename(int(no/2)))

            return segment_images, segment_flow

        return c3d_sampling()
    def sample_segment_test(self, s):
        """ Samples rgb and flow frame windows from a video segment s.
            Sampling 5 windows, equidistant along a video segment
            s = ["filename", start_frame, end_frame]"""

        def _path_to_dataset(flow):
            if flow:
                left = self.flow_data_path
            else:
                left = self.rgb_data_path
            right = "/frame_"
            numframe = 10
            return left, right, numframe

        #Optical flow stacking
        #returns u,v frames for EPIC Kitchens
        def flow_filename(frameno,num_stack=1):
            left, right,fill_frame = _path_to_dataset(True)
            left_frame = frameno - int((num_stack-1)/2)
            right_frame = frameno + int(num_stack/2)
            filename = []
            for no in range(left_frame, right_frame+1):
                filename.append(left + str(s[0]) + "/u" + right + str(no).zfill(fill_frame) + ".jpg")
                filename.append(left + str(s[0]) + "/v" + right + str(no).zfill(fill_frame) + ".jpg")
            return filename

        #return RGB frame for EPIC Kitchens
        def rgb_filename(frameno):
            left, right, fill_frame = _path_to_dataset(False)
            filename = left + str(s[0]) + right + str(frameno).zfill(fill_frame) + ".jpg"
            return filename

        def c3d_sampling():
            num_sample_frame = self.temporal_window
            half_sample_frame = int(self.temporal_window/2)
            segment_images = []
            segment_flow = []
            step = 2
            segment_start = int(s[1]) + (step*half_sample_frame)
            segment_end = int(s[2])+1 - (step*half_sample_frame)

            #if unable to keep all frames in sample inside segment, allow sampling outside of segment
            if segment_start >= segment_end:
                segment_start = int(s[1])#
                segment_end = int(s[2])
            # make sure sampling is not bellow frame 1
            if segment_start <= half_sample_frame*step+1:
                segment_start = half_sample_frame*step+2

            for center_frame in np.linspace(segment_start,segment_end,7,dtype=np.int32)[1:-1]:
                seg_f = []
                seg_i = []
                for no in range(center_frame - (step*half_sample_frame),center_frame + (step*half_sample_frame),step):
                    seg_f.append(flow_filename(int(no/2)))
                    seg_i.append(rgb_filename(no))
                segment_flow.append(seg_f)
                segment_images.append(seg_i)
            return segment_images, segment_flow

        return c3d_sampling()
    
    def nextBatchEval(self, sample_idx):
        """ Get next testing samples, return 5 equidistant frames along a action segment
            loops through datasets, randomly sampling data without replacement to add to batch. """
        # dataset_data, dataset_labels = self.dataset_data[test]
        # dataset_total = self.dataset_total[test]
        # dataset_data = dataset_data
        # dataset_labels = dataset_labels
        # dataset_total = dataset_total

        dataset_data = self.dataset_data
        dataset_labels = self.dataset_labels
        dataset_total = self.dataset_total

        batch_rgb = []
        batch_flow = []
        batch_labels = np.empty(shape=[0,self.num_labels])
        done = True
        if len(dataset_total) != 0:
            done = False

            #samples segment/frame
            # if len(dataset_total) > batch_size:
            #     sample_idx = np.random.choice(range(dataset_total.shape[0]), size=batch_size, replace=False)
            # else:
            #     sample_idx = range(dataset_total.shape[0])
            #     done = True

            # read and sample frames
            sample = dataset_data[dataset_total[sample_idx]]
            # read and sample frames
            sample = [self.sample_segment_test(sample)]
            sample_rgb,sample_flow = zip(*sample)

            sample_labels = dataset_labels[dataset_total[sample_idx]]

            sample_labels=np.expand_dims(sample_labels, axis=0)

            #remove frames/segments from epoch
            #self.dataset_total[test] = np.delete(dataset_total,sample_idx,axis=0)

            #create batch
            batch_labels = np.concatenate((sample_labels,batch_labels))
            batch_rgb = list(sample_rgb) + batch_rgb
            batch_flow = list(sample_flow) + batch_flow

        #print("batch_rgb",batch_rgb.shape)
        #reset dataset if all frames/segments have been evaluatated
        # if done:
        #     self.dataset_total[test] = np.arange(dataset_data.shape[0])

        batch_rgb = np.array(batch_rgb)
        #print("batch_rgb",batch_labels)
        # batch_flow = np.array(batch_flow)
        return done,batch_rgb,batch_flow,batch_labels
