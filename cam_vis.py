
import pyrealsense2 as rs 

import cv2 
import time 

import argparse

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from hmr2.models import HMR2, download_models, load_poselift, Poselift_CHECKPOINT, pose_lift
from hmr2.utils import recursive_to
from hmr2.configs import CACHE_DIR_4DHUMANS
from datetime import datetime
from hmr2.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)


import os
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio
#from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
# import mediapipe as mp
import neural_renderer as nr
from ultralytics import YOLO


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
YOLO_model = YOLO('yolov8n-pose.pt')  # load an official model
YOLO_model.predict(classes=0)


warnings.filterwarnings('ignore')

# log = get_pylogger(__name__)
# #/home/peter/Desktop/4D-Humans/logs/train/runs/pose_lift_trans_enc_dec/checkpoints/epoch=11-step=590000.ckpt
# download_models(CACHE_DIR_4DHUMANS)

# model_dir = "./logs/train/runs/pose_lift_trans_enc_dec/checkpoints/epoch=11-step=590000.ckpt"

# model = pose_lift.PoseLift.load_from_checkpoint(model_dir)

# # model, model_cfg = load_poselift()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = model.to(device)
# model.eval()



def main(): 

    width = 640
    height = 480

    pipeline = rs.pipeline()
    config = rs.config()
    #config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)

    # bounding_boxes = []
    # pose_keypoints_2d = []

    # ------------------default check point----------
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--out_folder', type=str, default='demo_pose_out', help='Output folder to save rendered results')
    parser.add_argument('--checkpoint', type=str, default=Poselift_CHECKPOINT, help='Path to pretrained model checkpoint')
    args = parser.parse_args()
    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_poselift(args.checkpoint)
    model.eval()
    # J_regressor = np.load('/media/maillab/peter/4D-Humans/data/J_regressor_coco.npy')

    # # ------------------setup model------------------
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()


    # # ------------------setup renderer------------------
    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    

    # # --------------make demo output folder---------------
    os.makedirs(args.out_folder, exist_ok=True)

    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


    previous_frame_time = 0
    new_frame_time = 0

    model_start_time = 0
    model_end_time = 0
    
    """video recording"""
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    file_name = str(current_datetime)

    expriment_video_dir = 'Expriment_Data_Collection_COKE_DiffCamLoc2_0820/Expriment_Videos'
    os.makedirs(expriment_video_dir, exist_ok=True)

    # Video Recording Section
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # scene_video_path = os.path.join(expriment_video_dir, f'Scence_Vid_{file_name}.avi')
    # scene_video_writer = cv2.VideoWriter(scene_video_path, fourcc, 12, (width, height), 1)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    scene_video_path = os.path.join(expriment_video_dir, f'Scence_Vid_{file_name}.avi')
    # fixed_fps = 30  # You can adjust this value
    scene_video_writer = cv2.VideoWriter(scene_video_path, fourcc, 8, (width, height))
    # scene_video_writer = cv2.VideoWriter(scene_video_path, fourcc, 12, (width, height))


    
    while True:
        # ------------------load detector------------------
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        yolo_out = YOLO_model(color_image, stream=True, verbose=True)

        box_image = color_image.copy()

        # output_test = model(test_dict)

        #1. Yolo outputs 2d pose, bbox
        all_verts = []
        all_cam_t = []
        if yolo_out is not None:
            for result in yolo_out:
                
                boxes = result.boxes
                mask = result.masks
                keypoints = result.keypoints
                # ---------------------------------
                # keypoints got two coordinated, xy and xyn, 
                # xy is corrdinated in the image
                # xyn is normalized coordinates with [0,1]
                # -------------------------------------
                
                probs = result.probs
                boxes = boxes.data.cpu().numpy() # [x1, y1, x2, y2], cof, class
                
                
             
                

                # print()
                
            
                if boxes is not None and len(boxes) > 0:
                    # find one instance


                    keypoints_xyn = keypoints.xyn.cpu().numpy()
                    keypoints_conf = keypoints.conf.cpu().numpy()
                    keypoints = keypoints.data.cpu().numpy() # [x, y, conf]

                    max_prob_index_box = np.argmax(boxes[:, -2])
                    max_prob_box = boxes[max_prob_index_box]  
                    
                    """box size"""
                    x1, y1, x2, y2 = max_prob_box[0], max_prob_box[1], max_prob_box[2], max_prob_box[3]
                    center_x = (x2-x1)//2
                    center_y = (y2-y1)//2
                    
                    box_center = np.array([center_x, center_y])
                    box_size  = (x2-x1) * (y2-y1)
                    
                    
                    img_size = 256
                    # label the person id
                    cls_id = max_prob_box[-1]
                    class_list = YOLO_model.model.names
                    class_name = class_list[cls_id]
                    
                    # get normalized xy coordinates
                    xy = keypoints_xyn[0, :, :2]  # Shape: (1, 17, 2)
                    xy = xy.reshape(1,17,2)
                    #rint(xyn)
                    # print(xyn.shape)
                    
                    conf = keypoints_conf[0, :]  # Shape: (17,)
                    conf = conf.reshape(1, 17, 1)  # Reshape to (1, 17, 1)

                    # Convert numpy arrays to torch tensors
                    xy_tensor = torch.from_numpy(xy).to(device)
                    confidence_tensor = torch.from_numpy(conf>0.7).float().to(device)
                    

                    # Concatenate xyn and confidence to get a tensor of shape (1, 17, 3)
                    combined_keypoints = torch.cat((xy_tensor, confidence_tensor), dim=-1)

                    # Now, combined_keypoints contains the normalized keypoints with their confidence values
                    # print("Combined Keypoints:", combined_keypoints.shape) (1,17,3)

                    # keypoints = torch.from_numpy(keypoints).to(device)
                    # keypoints = keypoints.reshape(1,17,3)
                    
                    
                    """try render"""            
                
                    test_dict = {
                    'input_keypoints_2d': combined_keypoints,
                    'box_center' : box_center,
                    }
                    # BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
                    # bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

           
                    
                    # ==========================================================================================================
                    cv2.putText(box_image, class_name,
                                (int(max_prob_box[0]), int(max_prob_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                        
                    cv2.rectangle(box_image, (int(max_prob_box[0]), int(max_prob_box[1])), (int(max_prob_box[2]), int(max_prob_box[3])), (0, 255, 0), 1)
                    
                    if keypoints is not None:
                        for keypoint in keypoints:
                            for point in keypoint:
                                x, y, prob = point[0], point[1], point[2]  # Extract x, y, and prob from keypoint
                                x, y = int(x), int(y)  # Convert to integers
                                #print(x, y, prob)

                                if max_prob_box[0] <= x <= max_prob_box[2] and max_prob_box[1] <= y <= max_prob_box[3]:
                                    # Draw the keypoint only if it is within the bounding box
                                    #pose_keypoints_2d = np.concatenate((pose_keypoints_2d, [[x, y]]), axis=0)
                                    cv2.circle(box_image, (x, y), 2, (0, 0, 255), -1)

                                # print(keypoints.shape)
                                # print(keypoints)
                                # 17 points
                                # -----------------------------boxes-----------------
                    
                                
                                            
                                pose_keypoints_2d = keypoints[0, :, :2]
                                # print(pose_keypoints_2d)
                                # print(pose_keypoints_2d.shape)
                                # print(pose_keypoints_2d)
                    # ===========================================================================================================
                else:
                    continue
                
        

            

        else:
            assert False, "No detections found by YOLOv8"

        # model_start_time = time.time()
        output_pose = model(test_dict)

        # model_end_time = time.time()

        # model_inference_time = model_end_time - model_start_time
        # print("Model inference time: ", model_inference_time)
        
        
        
        
        
        
        
        box_image_T =torch.from_numpy(color_image.copy().transpose(2, 0, 1)).float().to(device)
        #print(box_image_T.shape, "box_image_T shape")
        box_image = renderer(output_pose['pred_vertices'][0].detach().cpu().numpy(),
                                        output_pose['pred_cam_t'][0].detach().cpu().numpy(),
                                        box_image_T/255,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(0, 0, 0),
                                        )
        
        
        # # print(output_pose['pred_keypoints_3d'].shape) # torch.Size([1, 44, 3])
        # # print(output_pose['pred_vertices'].shape) # torch.Size([1, 6890, 3])
        # # print(J_regressor.shape,"J regressor shape coco") # (?,6890)
        
        # output_3d_kpts=J_regressor@output_pose['pred_vertices'].detach().cpu().numpy().squeeze()
        # output_3d_kpts=np.ascontiguousarray(output_3d_kpts.reshape(-1,3))#*640
        
        # # print(output_3d_kpts.shape, "output_3d_kpts shape")
        # # print(output_3d_kpts)
        
        # # print(output_3d_kpts[0,:], "output_3d_kpts 0") # (17, 3)
        # # # print(output_3d_kpts.shape, "output_3d_kpts shape") # (17, 3)
        # #print(output_pose.keys())
        # # print(output_pose['pred_cam'], output_pose['pred_cam_t'],"pred_cam_/t shape")
        # """find smpl coco regressor"""
        # # now do pnp
        # camera_matrix = np.array([[1300, 0, width/2.0], [0, 1300, height/2.0], [0, 0, 1]])
        # dist_coeffs = np.zeros((4,1))
        # pose_keypoints_2d=np.ascontiguousarray(pose_keypoints_2d.reshape(-1,2))
        # _,Rvec,Tvec=cv2.solvePnP(output_3d_kpts, pose_keypoints_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # # Rvec=output_pose['pred_cam'].detach().cpu().numpy().squeeze()
        # # Tvec=output_pose['pred_cam_t'].detach().cpu().numpy().squeeze()
        # verts_2d,_=cv2.projectPoints(output_pose['pred_vertices'].detach().cpu().numpy().squeeze(), Rvec, Tvec, camera_matrix, dist_coeffs)
        # verts_2d=np.ascontiguousarray(verts_2d.reshape(-1,2))
        # # verts_2d[:,1]=height-verts_2d[:,1]
        # # verts_2d[:,0]=width-verts_2d[:,0]
        # print(verts_2d.shape, "verts_2d shape")
        # # now draw verts!


        # for i in range(verts_2d.shape[0]):
        #     cv2.circle(box_image, (int(verts_2d[i, 0]), int(verts_2d[i, 1])), 1, (0, 0, 255), -1)
        
        # for i in range(verts_2d.shape[0]):
        #     cv2.circle(box_image, (int(verts_2d[i, 0]), int(verts_2d[i, 1])), 1, (0, 0, 255), -1)


        # do multipication with smpl regressor@pred_vertices
        #do pnp to get the camera pose

        # print(output_test)
        
        #Image presentation section
        # output_images = [(color_image, 'Camera Input'), 
        #                  (box_image, 'Bbox&Pose')
        #                  ]

        # for img, text in output_images:
        #             cv2.putText(img, text, (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
        #output_image = np.concatenate([img for img, _ in output_images], axis=1)   
        # cv2.imshow('Output', output_image)
        #cv2.imshow('Output', box_image)
        output_image = np.concatenate([color_image/255, box_image], axis=1)
        new_frame_time = time.time() 
        FPS = 1/(new_frame_time-previous_frame_time)
        # print("FPS: ", FPS)

        cv2.putText(output_image, "FPS: " + str(int(FPS)), (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow('Output', output_image)
        previous_frame_time = new_frame_time
        
        """recording"""
        # depth = output_image.dtype
        # print("The depth of the image is:", depth)
        output_image_normalized = cv2.normalize(output_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        output_image_uint8 = np.uint8(output_image_normalized)

        # Writing frame to video
        scene_video_writer.write(output_image_uint8)


        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            # out.release()
            break


    pipeline.stop()
    scene_video_writer.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()








