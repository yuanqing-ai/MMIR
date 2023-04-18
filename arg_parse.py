import argparse
import os

def parse_args():
    """
    Parse input arguments
    """


    
    parser = argparse.ArgumentParser(description='Train a MMIS network')
    parser.add_argument('--dataset', dest='dataset',
                        help="Comma seperated list of datasets",
                        default='None', type=str)
    parser.add_argument('--results_path', dest='results_path', 
                        help='Where to store the log files and saved models',default=None, type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    parser.add_argument('--rgb_data_path', help="rgb_data_path", default=None, type=str)
    parser.add_argument('--flow_data_path', help="flow_data_path", default=None, type=str)
    parser.add_argument('--train', help="Weither to train or evaluate (False)", default=None, type=bool)
    parser.add_argument('--lr', help="Initial Learning Rate", default=0.001, type=float)
    parser.add_argument('--batch_norm_update', help="Update rate of batch norm statistics", default=0.9, type=float)
    parser.add_argument('--num_gpus', help="number of gpus to run", default=4, type=int)
    parser.add_argument('--max_step', help="Number of batches to run", default=6000, type=int)
    parser.add_argument('--steps_before_update', help="number of steps to run before updating weights", default=1, type=int)
    parser.add_argument('--domain_mode', help="background only for dataset2", default=None, type=str)
    parser.add_argument('--lambda_in', help="grl hyperparameter", default=1.0, type=float)
    parser.add_argument('--self_lambda', help="weigthing of self supervised loss", default= 5.0, type=float)
    parser.add_argument('--unseen_dataset', help="Specify file path to unseen dataset folder", default=None, type=str)
    parser.add_argument('--num_labels', help="Total number of combined labels", default=8, type=int)
    parser.add_argument('--batch_size', help="Size of a batch", default=48, type=int)
    parser.add_argument('--seed', help="random seed", default=3, type=int)
    parser.add_argument('--epoch', help="epoch", default=100, type=int)
    parser.add_argument('--use_tfboard', help="use_tfboard", default=True, type=bool)

    parser.add_argument('--synchronised', help="Weither to synchronise flow and rgb", default=None, type=bool)
    parser.add_argument('--modality', help="rgb, flow or joint (default: joint)", default='rgb', type=str)
    parser.add_argument('--temporal_window', help="i3d temporal window", default=16, type=int)
    parser.add_argument('--aux_classifier', help="2 classifiers", default=None, type=bool)
    #parser.add_argument('--pred_synch', help="Predict if modalities are synchronised", default=None, type=bool)
    parser.add_argument('--features', help="Weither to produce features of evalutate", default=None, type=bool)
    parser.add_argument('--feature_path', help="path to store features", default=None, type=str)
    parser.add_argument('--eval_train', help="Weither to evaludate training example rather than test", default=None, type=bool)
    parser.add_argument('--modelnum', help="model number to restore for testing", default=None, type=int)
    parser.add_argument('--restore_model_rgb', help="Load these weights excluding Logits", default=None, type=str)
    parser.add_argument('--restore_model_flow', help="Load these weights excluding Logits", default=None, type=str)
    parser.add_argument('--restore_model_joint', help="Load these weights excluding Logits", default=None, type=str)
    parser.add_argument('--restore_mode', help="pretrain (for base netwrok without logits),"
                                              " model (restore base model with classification logits) "
                                              " or continue (restore everything)", default=None, type=str)


    parser.add_argument('--trained_path', help="model_trained without DA", default=None, type=str)
    parser.add_argument('--ts_flow', help="ts_flow", default=0.5, type=float)

    parser.add_argument('--tt_flow', help="tt_flow", default=0.5, type=float)


    parser.add_argument('--ts_RGB', help="ts_RGB", default=0.5, type=float)

    parser.add_argument('--tt_RGB', help="tt_RGB", default=0.5, type=float)

    parser.add_argument('--S_agent_flow', help="S_agent_flow", default=False, type=bool)
    parser.add_argument('--T_agent_flow', help="T_agent_flow", default=False, type=bool)

    parser.add_argument('--S_agent_RGB', help="S_agent_RGB", default=False, type=bool)

    parser.add_argument('--T_agent_RGB', help="T_agent_RGB", default=False, type=bool)
    parser.add_argument('--restoring', help="restoring", default=False, type=bool)

    
    
    parser.add_argument('--select_num', help="select_num", default=1, type=int)
    parser.add_argument('--candidate_num', help="candidate_num", default=5, type=int)

    parser.add_argument('--epsilon_decay', help="epsilon_decay", default=24000, type=int)

    parser.add_argument('--REPLAY_MEMORY', help="REPLAY_MEMORY", default=200, type=int)

    parser.add_argument('--batch_dqn', help="batch_dqn", default=64, type=int)

    parser.add_argument('--step', help="step", default=100, type=int)

    parser.add_argument('--replace_target_iter', help="replace_target_iter", default=48, type=int)
    parser.add_argument('--pred_synch',help="synch class",default=None,type=bool)



    parser.add_argument('--epsilon_start', help="start of epsilon", default=0.9, type=float)
    parser.add_argument('--epsilon_final', help="end of epsilon", default=0.01, type=float)



    args = parser.parse_args()
    #print(args)


    source_domain = os.path.basename(args.dataset)
    target_domain = os.path.basename(args.unseen_dataset)
    train_dir = args.results_path + "/saved_model_" + source_domain + "_" + target_domain + "_" + str(args.lr) + "_" + str(
        args.batch_norm_update)
    # if not os.path.exists(train_dir):
    #     os.makedirs(train_dir)
    results_dir = args.results_path + "/results_" + source_domain + "_" + target_domain + "_" + str(args.lr) + "_" + str(
        args.batch_norm_update)


    return args, train_dir, results_dir