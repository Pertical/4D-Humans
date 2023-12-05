
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

# from hmr2.datasets.utils import expand_bbox_to_aspect_ratio
#from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD

from ultralytics import YOLO


YOLO_model = YOLO('yolov8n-pose.pt')  # load an official model
YOLO_model.predict(classes=0)


warnings.filterwarnings('ignore')

log = get_pylogger(__name__)




def main(): 

    width = 640
    height = 480

    pipeline = rs.pipeline()
    config = rs.config()
    #config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)

    # cfg = Human4DConfig()
    # hmr_traker = HMR2_4dhuman(cfg)

    bounding_boxes = []
    pose_keypoints_2d = []

    # # ------------------default check point----------
    # parser = argparse.ArgumentParser(description='HMR2 demo code')
    # parser.add_argument('--out_folder', type=str, default='demo_pose_out', help='Output folder to save rendered results')
    # parser.add_argument('--checkpoint', type=str, default=Poselift_CHECKPOINT, help='Path to pretrained model checkpoint')
    # args = parser.parse_args()
    # # Download and load checkpoints
    # download_models(CACHE_DIR_4DHUMANS)
    # model, model_cfg = load_poselift(args.checkpoint)


    # # ------------------setup model------------------
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()


    # # ------------------setup renderer------------------
    # renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # # --------------make demo output folder---------------
    # os.makedirs(args.out_folder, exist_ok=True)

    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


    while True:
        # ------------------load detector------------------
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        yolo_out = YOLO_model(color_image, stream=True, verbose=False)

        box_image = color_image.copy()


        #1. Yolo outputs 2d pose, bbox

        if yolo_out is not None:
            for result in yolo_out:
                
                boxes = result.boxes
                mask = result.masks
                keypoints = result.keypoints
                probs = result.probs
                
                boxes = boxes.data.cpu().numpy()
                keypoints = keypoints.data.cpu().numpy()

                if boxes is not None and len(boxes) > 0:
                    max_prob_index_box = np.argmax(boxes[:, -2])
                    max_prob_box = boxes[max_prob_index_box]


                    #print(max_prob_box)

                    cls_id = max_prob_box[-1]

                    class_list = YOLO_model.model.names
                    class_name = class_list[cls_id]


                    cv2.putText(box_image, class_name,
                                (int(max_prob_box[0]), int(max_prob_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                        
                    cv2.rectangle(box_image, (int(max_prob_box[0]), int(max_prob_box[1])), (int(max_prob_box[2]), int(max_prob_box[3])), (0, 255, 0), 1)
                    
                    if keypoints is not None:
                        for keypoint in keypoints:
                            for point in keypoint:
                                x, y, prob = point[0], point[1], point[2]  # Extract x, y, and prob from keypoint
                                x, y = int(x), int(y)  # Convert to integers
                                # print(x, y, prob)

                                if max_prob_box[0] <= x <= max_prob_box[2] and max_prob_box[1] <= y <= max_prob_box[3]:
                                    # Draw the keypoint only if it is within the bounding box
                                    pose_keypoints_2d.append([x, y])
                                    cv2.circle(box_image, (x, y), 2, (0, 0, 255), -1)

                else:
                    continue

                
            
            bounding_boxes = np.asarray(bounding_boxes)


            # dataset = ViTDetDataset(model.cfg, color_frame, bounding_boxes)
            # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            # for batch in dataloader:
            #     batch = recursive_to(batch, device)
            #     with torch.no_grad():
            #         out = model(batch)

            # batch_size = batch['img'].shape[0]
            # for n in range(batch_size):
            #     # Get filename from path img_path
            #     input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            #     input_patch = input_patch.permute(1,2,0).numpy()

            #     regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #                             out['pred_cam_t'][n].detach().cpu().numpy(),
            #                             batch['img'][n],
            #                             mesh_base_color=LIGHT_BLUE,
            #                             scene_bg_color=(1, 1, 1),
            #                             )

            #     final_img = np.concatenate([input_patch, regression_img], axis=1)
        


            

        else:
            assert False, "No detections found by YOLOv8"

            

        #2. YOLO ouputs into HMR2_4dhuman

        #3. HMR2_4dhuman outputs smpl pose, bbox, 3d pose

        #4. cv2 video show 



        #Image presentation section
        output_images = [(color_image, 'Camera Input'), 
                         (box_image, 'Bbox&Pose')
                         ]

        for img, text in output_images:
                    cv2.putText(img, text, (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
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








