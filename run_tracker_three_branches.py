import os
import sys
import argparse
import pdb
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker
import torch
import random
import numpy as np

def run_tracker_three_branches(args, tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
        # add by liulei
        guide: if use guide module
        quality_aware: if use quality aware module
    """

    visdom_info = {} if visdom_info is None else visdom_info
    dataset = get_dataset(dataset_name)
    #pdb.set_trace()
    if sequence is not None:
        dataset = [dataset[sequence]]
    trackers = [Tracker(tracker_name, tracker_param, run_id)]
    run_dataset(args, dataset, trackers, debug, threads, visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='dimp', help='Name of tracking method.') # atom, base, dimp, eco
    parser.add_argument('--tracker_param', type=str, default='dimp50_rgbt_three_branches', help='Name of parameter file.') 
    # [atom] atom_gmm_sampl, atom_prob_ml, default, default_vot, multiscale_no_iounet, 
    # [dimp] dimp18, dimp18_vot, dimp50, dimp50_vot, dimp50_vot19, prdimp18, prdimp50, super_dimp, super_dimp_rgbt, prdimp50_rgbt, super_dimp_rgbt_three_branch
    # [eco]  default, mobile3
    parser.add_argument('--runid', type=int, default=20211120, help='The run id.') 
    parser.add_argument('--dataset_name', type=str, default='rgbt234', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gtot, gotv, lasot, gtot_v, gtot, rgbt210, rgbt234, lasher, lashertestingset).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    # add by liulei
    parser.add_argument('--guide', type=bool, default=False, help='Use guide module.')
    parser.add_argument('--guide_module_name', type=str, default='cat_v6', help='guide module name.') 
    parser.add_argument('--guide_module_path_01', type=str, default='DiMPnet_RGBT_ep0079.pth.tar', help='guide module 01 path.')
    parser.add_argument('--guide_module_path_10', type=str, default='DiMPnet_RGBT_ep0079.pth.tar', help='guide module 10 path.')
    parser.add_argument('--norm', type=bool, default=False, help='norm qam score')
    parser.add_argument('--threshold', type=float, default=0.00, help='qam_score_threshold')
    parser.add_argument('--max_score_thr', type=float, default=0.00, help='max_score_thr')
    parser.add_argument('--quality_aware', type=bool, default=False, help='Use quality_aware module.')
    parser.add_argument('--quality_aware_module_name', type=str, default='qam_pt_10', help='quality aware module name.')  
    parser.add_argument('--quality_aware_module_path', type=str, default='5573.pth', help='quality aware module path.') 
    parser.add_argument('--random_label', type=bool, default=False, help='Use random_label rather than quality_aware module or human_label.')    
    parser.add_argument('--human_label', type=str, default='predict', help='Use quality label/challenge label/predict_score.') 
    parser.add_argument('--first_frame_guide', type=bool, default=False, help='use guide module in first frame.')     
    parser.add_argument('--save_qam_score', type=bool, default=False, help='Save QAM Score.')
    parser.add_argument('--vis_qam_score', type=bool, default=False, help='Vis QAM Score.')
    parser.add_argument('--three_branch', type=bool, default=True, help='Generate quality label for datasets.')
    parser.add_argument('--save_three_results', type=bool, default=True, help='Save tracking results of three branch.')
    parser.add_argument('--save_feature_map', type=bool, default=False, help='Save features for training QAM.')
    args = parser.parse_args()
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    print(args)
    run_tracker_three_branches(args, args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug, args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port})


if __name__ == '__main__':
    main()
