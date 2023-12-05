
import pyrealsense2 as rs 

import cv2 

import argparse

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from hmr2.models import HMR2, download_models, load_poselift, Poselift_CHECKPOINT, pose_lift
from hmr2.utils import recursive_to
from hmr2.configs import CACHE_DIR_4DHUMANS


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
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
import mediapipe as mp
import neural_renderer as nr
from ultralytics import YOLO



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

    # # ------------------setup model------------------
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()


    # # ------------------setup renderer------------------
    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    

    # # --------------make demo output folder---------------
    os.makedirs(args.out_folder, exist_ok=True)

    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
    

    while True:
        # ------------------load detector------------------
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        yolo_out = YOLO_model(color_image, stream=True, verbose=False, imgsz=640, conf=0.3,classes=0,save=False,half=True)

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
                
                # print(boxes)
                # print("boxes shape:", boxes.shape, "\n")
                
                keypoints_xyn = keypoints.xyn.cpu().numpy()
                # keypoints = keypoints.data.cpu().numpy()
                # print(keypoints)
                # print("keypoint shape:", keypoints.shape, "\n")
                
                keypoints_conf = keypoints.conf.cpu().numpy()
                # print(keypoints_conf.shape)    
                               
                if boxes is not None and len(boxes) > 0:
                    # find one instance
                    max_prob_index_box = np.argmax(boxes[:, -2])
                    max_prob_box = boxes[max_prob_index_box]   
                    
                    # label the person id
                    cls_id = max_prob_box[-1]
                    class_list = YOLO_model.model.names
                    class_name = class_list[cls_id]
                    
                    # get normalized xy coordinates
                    xyn = keypoints_xyn[0, :, :2]  # Shape: (1, 17, 2)
                    xyn = xyn.reshape(1,17,2)
                    # print(xyn)
                    # print(xyn.shape)
                    
                    confidence = keypoints_conf[0, :]  # Shape: (17,)
                    confidence = confidence.reshape(1, 17, 1)  # Reshape to (1, 17, 1)

                    # Convert numpy arrays to torch tensors
                    xyn_tensor = torch.from_numpy(xyn).to(device)
                    confidence_tensor = torch.from_numpy(confidence).to(device)

                    # Concatenate xyn and confidence to get a tensor of shape (1, 17, 3)
                    combined_keypoints = torch.cat((xyn_tensor, confidence_tensor), dim=-1)

                    # Now, combined_keypoints contains the normalized keypoints with their confidence values
                    # print("Combined Keypoints:", combined_keypoints.shape) (1,17,3)
                    
                    
                    
                    
                    
                    # center_x = 
                    
                    # # ------------------img--------------------
                    # cvimg = self.img_cv2.copy()
                    # if True:
                    #     # Blur image to avoid aliasing artifacts
                    #     downsampling_factor = ((bbox_size*1.0) / patch_width)
                    #     print(f'{downsampling_factor=}')
                    #     downsampling_factor = downsampling_factor / 2.0
                    #     if downsampling_factor > 1.1:
                    #         cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


                    # img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                    #                                             center_x, center_y,
                    #                                             bbox_size, bbox_size,
                    #                                             patch_width, patch_height,
                    #                                             False, 1.0, 0,
                    #                                             border_mode=cv2.BORDER_CONSTANT)
                    
                    # -----------------------------boxes-----------------
                    
                    x1, y1, x2, y2 = boxes[0,0], boxes[0,1], boxes[0,2], boxes[0,3]
                    box_center = (x2-x1)//2
                    
                    
                    
                    
                    test_dict = {
                    'input_keypoints_2d': combined_keypoints}
                    # print(type(item))
                    # boxes = boxes.data.cpu().numpy()
                    # keypoints = keypoints.data.cpu().numpy()
                    # print(boxes)
                    
                    
                    
                    
                    # ==========================================================================================================
                    # cv2.putText(box_image, class_name,
                    #             (int(max_prob_box[0]), int(max_prob_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                        
                    # cv2.rectangle(box_image, (int(max_prob_box[0]), int(max_prob_box[1])), (int(max_prob_box[2]), int(max_prob_box[3])), (0, 255, 0), 1)
                    
                    # if keypoints is not None:
                    #     for keypoint in keypoints:
                    #         for point in keypoint:
                                # x, y, prob = point[0], point[1], point[2]  # Extract x, y, and prob from keypoint
                                # x, y = int(x), int(y)  # Convert to integers
                                # print(x, y, prob)

                                # if max_prob_box[0] <= x <= max_prob_box[2] and max_prob_box[1] <= y <= max_prob_box[3]:
                                #     # Draw the keypoint only if it is within the bounding box
                                #     #pose_keypoints_2d = np.concatenate((pose_keypoints_2d, [[x, y]]), axis=0)
                                #     cv2.circle(box_image, (x, y), 2, (0, 0, 255), -1)

                                # print(keypoints.shape)
                                # print(keypoints)
                                #17 points
                                
                                # pose_keypoints_2d = keypoints[0, :, :2]
                                # print(pose_keypoints_2d)
                                # print(pose_keypoints_2d.shape)
                                # print(pose_keypoints_2d)
                    # ===========================================================================================================
                else:
                    continue
                
                
                # max_prob_index_box = np.argmax(boxes[:, -2])
                # max_prob_box = boxes[max_prob_index_box]
                
           
                # ------------------------------------------------------------
                # boxes = boxes.data.cpu().numpy()
                # keypoints = keypoints.data.cpu().numpy()
                # for box in boxes:
                #     bounding_boxes.append(box)
                #     #print(box)
                #     cls_id = box[-1]

                #     class_list = YOLO_model.model.names
                #     class_name = class_list[cls_id]

                #     cv2.putText(box_image, class_name,
                #             (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                    
                #     cv2.rectangle(box_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

                # for keypoint in keypoints:
                #     for point in keypoint:
                #         x, y, prob = point[0], point[1], point[2]  # Extract x, y, and prob from keypoint
                #         x, y = int(x), int(y)  # Convert to integers
                #         # print(x, y, prob)

                #         if prob > 0.3:
                #             pose_keypoints_2d.append([x, y])
                #             cv2.circle(box_image, (x, y), 2, (0, 0, 255), -1) 
        
            

        else:
            assert False, "No detections found by YOLOv8"

        output_test = model(test_dict)
        print(output_test['pred_keypoints_3d'])
        print(output_test['pred_vertices'])
        # print(output_test)
        
        #Image presentation section
        output_images = [(color_image, 'Camera Input'), 
                         (box_image, 'Bbox&Pose')
                         ]

        # for img, text in output_images:
        #             cv2.putText(img, text, (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
        output_image = np.concatenate([img for img, _ in output_images], axis=1)   
        cv2.imshow('Output', output_image)


        # humr_result = hmr_traker.get_detections(color_image, 'frame_name', 0)
        # hmr_traker.visualize(color_image, humr_result)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            # out.release()
            break


    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()








