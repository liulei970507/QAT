import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from pytracking.evaluation import Sequence, Tracker
from ltr.data.image_loader import imwrite_indexed
import pdb

def _save_tracker_output(dataset_name, seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
        os.makedirs(os.path.join(tracker.results_dir, dataset_name))

    base_results_path = os.path.join(tracker.results_dir, dataset_name, '{}_{}_{}'.format(tracker.parameter_name, tracker.run_id, seq.name))
    #print('base_results_path',base_results_path)
    segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    #frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict
    
    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

def _save_tracker_iou(dataset_name, seq: Sequence, tracker: Tracker, iou, iou_v, iou_i):
    """Saves the iou of the tracker."""
    if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
        os.makedirs(os.path.join(tracker.results_dir, dataset_name))
    base_results_path = os.path.join(tracker.results_dir, dataset_name, '{}_iou'.format(seq.name))
    def save_bb(file, data):
        tracked_bb = np.array(data)#.astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%f')
    data = []
    for i in range(len(iou)):
        data.append([i, iou[i], iou_v[i], iou_i[i]])
    # pdb.set_trace()
    bbox_file = '{}.txt'.format(base_results_path)
    save_bb(bbox_file, data)

def _save_tracker_qam_score(dataset_name, seq: Sequence, tracker: Tracker, qam_score_layer2, qam_score_layer3):
    """Saves the iou of the tracker."""
    if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
        os.makedirs(os.path.join(tracker.results_dir, dataset_name))
    base_results_path_layer2 = os.path.join(tracker.results_dir, dataset_name, '{}_qam_score_layer2'.format(seq.name))
    base_results_path_layer3 = os.path.join(tracker.results_dir, dataset_name, '{}_qam_score_layer3'.format(seq.name))
    def save_bb(file, data):
        tracked_bb = np.array(data)#.astype(int)
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%f')
    data_layer2 = []
    data_layer3 = []
    for i in range(len(qam_score_layer2)):
        # pdb.set_trace()
        # data_layer2.append([qam_score_layer2[i][0], qam_score_layer2[i][1]])
        # data_layer3.append([qam_score_layer3[i][0], qam_score_layer3[i][1]])
        data_layer2.append([qam_score_layer2[i].squeeze()[0],qam_score_layer2[i].squeeze()[1]])
        data_layer3.append([qam_score_layer3[i].squeeze()[0],qam_score_layer3[i].squeeze()[1]])
    # pdb.set_trace()
    bbox_file_layer2 = '{}.txt'.format(base_results_path_layer2)
    save_bb(bbox_file_layer2, data_layer2)
    bbox_file_layer3 = '{}.txt'.format(base_results_path_layer3)
    save_bb(bbox_file_layer3, data_layer3)

# def _save_tracker_feature_map(dataset_name, seq: Sequence, tracker: Tracker, features_layer2, features_layer3):
#     """Saves the iou of the tracker."""
#     if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
#         os.makedirs(os.path.join(tracker.results_dir, dataset_name))
#     base_results_path = os.path.join(tracker.results_dir, dataset_name, '{}_features'.format(seq.name))
#     def save_bb(file, data):
#         tracked_result = np.array(data)#.astype(int)
#         np.save(file, tracked_result)
#     data = []
#     for i in range(len(features_layer2)):
#         data.append([features_layer2[i], features_layer3[i]])
#     # pdb.set_trace()
#     result_file = '{}.npy'.format(base_results_path)
#     save_bb(result_file, data)

def _save_tracker_feature_map(dataset_name, seq: Sequence, tracker: Tracker, features_layer3):
    """Saves the iou of the tracker."""
    if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
        os.makedirs(os.path.join(tracker.results_dir, dataset_name))
    base_results_path = os.path.join(tracker.results_dir, dataset_name, '{}_features'.format(seq.name))
    def save_bb(file, data):
        tracked_result = np.array(data)#.astype(int)
        np.save(file, tracked_result)
    data = []
    for i in range(len(features_layer3)):
        data.append(features_layer3[i])
    # pdb.set_trace()
    result_file = '{}.npy'.format(base_results_path)
    save_bb(result_file, data)

def _save_tracker_output_three_branch(dataset_name, seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(os.path.join(tracker.results_dir, dataset_name)):
        os.makedirs(os.path.join(tracker.results_dir, dataset_name))
    base_results_path_v = os.path.join(tracker.results_dir, dataset_name, '{}_tracker_results_v'.format(seq.name))
    base_results_path_i = os.path.join(tracker.results_dir, dataset_name, '{}_tracker_results_i'.format(seq.name))
    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict
    
    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox_v':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path_v, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path_v)
                save_bb(bbox_file, data)

        if key == 'target_bbox_i':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path_i, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path_i)
                save_bb(bbox_file, data)

def run_sequence(args, seq: Sequence, tracker: Tracker, debug=False, visdom_info=None):
    # add by liulei
    # guide: if use guide module
    # quality_aware: if use quality aware module
    """Runs a tracker on a sequence."""
    def _results_exist():
        if seq.object_ids is None:
            bbox_file = '{}/{}_{}_{}.txt'.format(tracker.results_dir, args.dataset_name, tracker.parameter_name, tracker.run_id, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0
            
    def overlap_ratio(rect1, rect2):
        '''
        Compute overlap ratio between two rects
        - rect: 1d array of [x,y,w,h] or
                2d array of N x [x,y,w,h]
        '''
    
        if rect1.ndim==1:
            rect1 = rect1[None,:]
        if rect2.ndim==1:
            rect2 = rect2[None,:]
    
        left = np.maximum(rect1[:,0], rect2[:,0])
        right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
        top = np.maximum(rect1[:,1], rect2[:,1])
        bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])
    
        intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
        union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return iou

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('{} FPS: {}'.format(seq.name, -1))
        return 
    if debug:
        output = tracker.run_sequence(args, seq, debug=debug, visdom_info=visdom_info)
    else:
        output = tracker.run_sequence(args, seq, debug=debug, visdom_info=visdom_info)
    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])
    iou = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox'])) # iou of each frame between gt and bbox of this seq
    ave_iou = iou.sum()/len(iou)
    fps = num_frames / exec_time
    if args.three_branch:
        iou_v = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox_v'])) # iou of each frame between gt and bbox of this seq
        ave_iou_v = iou_v.sum()/len(iou_v)
        iou_i = overlap_ratio(np.array(seq.ground_truth_rect), np.array(output['target_bbox_i'])) # iou of each frame between gt and bbox of this seq
        ave_iou_i = iou_i.sum()/len(iou_i)
    #print('IOU: {} FPS: {}'.format(ave_iou, fps))
    if not debug:
        _save_tracker_output(args.dataset_name, seq, tracker, output)
    if args.three_branch:
        _save_tracker_iou(args.dataset_name, seq, tracker, iou, iou_v, iou_i)
        if args.save_feature_map:
            # _save_tracker_feature_map(args.dataset_name, seq, tracker, output['features_layer2'], output['features_layer3'])
            _save_tracker_feature_map(args.dataset_name, seq, tracker, output['features_layer3'])
        if args.save_three_results:
            _save_tracker_output_three_branch(args.dataset_name, seq, tracker, output)
        return iou.tolist(),ave_iou, ave_iou_v, ave_iou_i, fps
    else:
        # _save_tracker_qam_score(args.dataset_name, seq, tracker, output['qam_score_layer2'], output['qam_score_layer3'])
        return iou.tolist(),ave_iou, fps
    # else:
    #     return iou.tolist(),ave_iou, fps

def run_dataset(args, dataset, trackers, debug=False, threads=0, visdom_info=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
        # add by liulei
        guide: if use guide module
        quality_aware: if use quality aware module
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    visdom_info = {} if visdom_info is None else visdom_info
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        iou_list = []
        iou_list_v = []
        iou_list_i = []
        for seq in dataset:
            for tracker_info in trackers:
                # try:
                    if args.three_branch:
                        # import pdb
                        # pdb.set_trace()
                        # print(seq.name)
                        # if os.path.exists('/data/liulei/pytracking/pytracking/tracking_results/dimp/super_dimp_rgbt_three_branch_20211025/lasher/super_dimp_rgbt_three_branch_20211025_'+seq.name+'.txt'):
                        #     print('continue')
                        #     continue
                        iou, ave_iou, ave_iou_v, ave_iou_i, fps = run_sequence(args, seq, tracker_info, debug=debug, visdom_info=visdom_info)
                        iou_list = iou_list+iou
                        iou_list_v.append(ave_iou_v)
                        iou_list_i.append(ave_iou_i)
                        print('Tracker: {} {} {}, Sequence: {}, IOU: {} IOU_V: {} IOU_I: {} FPS: {}, total mIoU: {}'.format(tracker_info.name, tracker_info.parameter_name, tracker_info.run_id, seq.name, ave_iou, ave_iou_v, ave_iou_i, fps, sum(iou_list)/len(iou_list)))
                    else:
                        # print(seq.name)
                        # if os.path.exists('/home/liulei/pytracking/pytracking/tracking_results/dimp/super_dimp_rgbt_20210602012300501/lashertestingset/super_dimp_rgbt_20210602012300501_'+seq.name+'.txt'):
                        #     print('continue')
                        #     continue
                        iou, ave_iou, fps = run_sequence(args, seq, tracker_info, debug=debug, visdom_info=visdom_info)
                        iou_list = iou_list+iou
                        print('Tracker: {} {} {}, Sequence: {}, IOU: {} FPS: {}, total mIoU: {}'.format(tracker_info.name, tracker_info.parameter_name, tracker_info.run_id, seq.name, ave_iou, fps, sum(iou_list)/len(iou_list)))
                # except Exception as e:
                #     print('running.py line173 Exception:',e)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
