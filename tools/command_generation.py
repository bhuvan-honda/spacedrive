# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------

import pickle
import numpy as np


from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
def command_generation(gt_traj_path, new_gt_traj_path):
    ''' 
    Generate 'gt_planning_command' based on the heading degree of 'gt_planning'
    Command definition:
        -0.4 - 0.4: strictly forward -> 0
        -2 - -0.4 to 0.4 - 2: forward -> 1
        2 - 10: slightly right -> 2
        10 - 20: right -> 3
        >20: sharp right -> 4
        -2 - -10: slightly left -> 5
        -10 - -20: left -> 6
        <-20: sharp left -> 7
    
    '''
    print('reading pickle')
    anno = pickle.load(open(gt_traj_path, 'rb'))
    print(anno.keys())
    key_infos = anno['infos']
    print('reading pickle finished')

    for i in range(len(key_infos)):
        key_info = key_infos[i]
        planning_traj = key_info['gt_planning'][0]

        angle = np.arctan2(planning_traj[-1, 1], planning_traj[-1, 0]) * 180 / np.pi
            
        if np.linalg.norm(planning_traj[-1]) < 0.5:
            angle = 0

        if angle >= -0.4 and angle <= 0.4:
            command = 0
            gt_planning_command_desc = "The ego vehicle is moving strictly forward. "  
        elif (angle > 0.4 and angle <= 2) or (angle < -0.4 and angle >= -2):
            command = 1
            gt_planning_command_desc = "The ego vehicle is moving forward. "  
        elif angle > 2 and angle <=10:
            command = 2
            gt_planning_command_desc = "The ego vehicle is moving slightly right. "  
        elif angle >10 and angle <=20:
            command = 3
            gt_planning_command_desc = "The ego vehicle is turning right. "  
        elif angle >20:
            command = 4
            gt_planning_command_desc = "The ego vehicle is turning sharp right. "  
        elif angle < -2 and angle >= -10:
            command = 5
            gt_planning_command_desc = "The ego vehicle is moving slightly left. "  
        elif angle < -10 and angle >= -20:
            command = 6
            gt_planning_command_desc = "The ego vehicle is turning left. "  
        elif angle < -20:
            command = 7
            gt_planning_command_desc = "The ego vehicle is turning sharp left. "  
        
        key_infos[i]['gt_planning_command'] = command
        key_infos[i]['gt_planning_command_desc'] = gt_planning_command_desc

    # save the new key_infos with gt_planning_command
    print('writing pickle')
    anno_modified = {}
    anno_modified['infos'] = key_infos
    for key in anno.keys():
        if key != 'infos':
            anno_modified[key] = anno[key]
    pickle.dump(anno_modified, open(new_gt_traj_path, 'wb'))
    print('writing pickle finished')




if __name__ == "__main__":

    dataset_path = 'data/nuscenes/' # '../../data/nuscenes/'

    gt_traj_path_train = dataset_path + 'nuscenes2d_ego_temporal_infos_train.pkl'
    new_gt_traj_path_train = dataset_path + 'nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl'
 
    gt_traj_path_val = dataset_path + 'nuscenes2d_ego_temporal_infos_val.pkl'
    new_gt_traj_path_val = dataset_path + 'nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl'

    print('processing train')
    command_generation(gt_traj_path_train, new_gt_traj_path_train)

    print('processing val')
    command_generation(gt_traj_path_val, new_gt_traj_path_val)
