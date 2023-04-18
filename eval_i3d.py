








val_data_T = Epic(self.num_labels, self.unseen_datasets,
                                    temporal_window=self.FLAGS.temporal_window,
                                    rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
                                   synchronised=self.FLAGS.synchronised,test=True, random_sync=self.FLAGS.pred_synch)

valoader_t = DataLoader(val_data_T,batch_size=batch_per_gpu,shuffle=False,num_workers=2)



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

self.model.load_state_dict(torch.load(self.save_model+'/'+str(steps).zfill(6)+'.pt'))



val_iter_s = iter(valoader_s)
inputs = val_iter_s
valaccuracy, average_class = evaluate(self.model, self.FLAGS, inputs, lin, test=True, extra_info=True)



