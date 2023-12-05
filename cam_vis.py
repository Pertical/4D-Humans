
import pyrealsense2 as rs 

import cv2 

import argparse

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
<<<<<<< HEAD
=======
from hmr2.models import HMR2, download_models, load_poselift, Poselift_CHECKPOINT, pose_lift
from hmr2.utils import recursive_to
from hmr2.configs import CACHE_DIR_4DHUMANS
>>>>>>> origin/main


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
<<<<<<< HEAD

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

=======
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
>>>>>>> origin/main

from ultralytics import YOLO


YOLO_model = YOLO('yolov8n-pose.pt')  # load an official model
YOLO_model.predict(classes=0)


warnings.filterwarnings('ignore')

log = get_pylogger(__name__)


<<<<<<< HEAD
class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

    def forward(self, x):
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                        face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                        # textures=texture_atlas_rgb,
                                        mode='depth',
                                        K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam':  model_out['pred_cam'],
        }
        return out
    

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    
    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, 
            ground_truth_track_id, ground_truth_annotations
        ) =  super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )

    def visualize(self, image, detections):
        (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        ) = detections

        # Visualize
        vis_image = image.copy()
        for i in range(len(pred_bbox)):
            bbox = pred_bbox[i]
            bbox_padded = pred_bbox_padded[i]
            mask = pred_masks[i]
            score = pred_scores[i]
            cls = pred_classes[i]

            # Draw bbox
            bbox = bbox.astype(np.int32)
            bbox_padded = bbox_padded.astype(np.int32)
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(vis_image, (bbox_padded[0], bbox_padded[1]), (bbox_padded[2], bbox_padded[3]), (0, 0, 255), 2)

            # Draw mask
            mask = (mask > 0.5).astype(np.uint8)
            mask = np.stack([mask, mask, mask], axis=2)
            vis_image = np.where(mask == 1, vis_image, vis_image * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

            # Draw score
            cv2.putText(vis_image, f'{score:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw class
            cv2.putText(vis_image, f'{cls}', (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Vis Image', vis_image)
        cv2.waitKey(1)

=======
>>>>>>> origin/main


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

<<<<<<< HEAD

    while True:

=======
    # ------------------default check point----------
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--out_folder', type=str, default='demo_pose_out', help='Output folder to save rendered results')
    parser.add_argument('--checkpoint', type=str, default=Poselift_CHECKPOINT, help='Path to pretrained model checkpoint')
    args = parser.parse_args()
    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_poselift(args.checkpoint)


    # ------------------setup model------------------
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()


    # ------------------setup renderer------------------
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # --------------make demo output folder---------------
    os.makedirs(args.out_folder, exist_ok=True)

    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


    while True:
        # ------------------load detector------------------
>>>>>>> origin/main
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        yolo_out = YOLO_model(color_image, stream=True, verbose=False)

        box_image = color_image.copy()


        #1. Yolo outputs 2d pose, bbox

        if yolo_out is not None:
<<<<<<< HEAD
=======

>>>>>>> origin/main
            for result in yolo_out:
                
                boxes = result.boxes
                mask = result.masks
                keypoints = result.keypoints
                probs = result.probs
                
                boxes = boxes.data.cpu().numpy()
                keypoints = keypoints.data.cpu().numpy()

                #print(boxes.shape)

<<<<<<< HEAD
                max_prob_index = np.argmax(boxes[:, -2])
                max_prob_box = boxes[max_prob_index]

                #print(max_prob_box)

                cls_id = max_prob_box[-1]

                class_list = YOLO_model.model.names
                class_name = class_list[cls_id]


                cv2.putText(box_image, class_name,
                            (int(max_prob_box[0]), int(max_prob_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                    
                cv2.rectangle(box_image, (int(max_prob_box[0]), int(max_prob_box[1])), (int(max_prob_box[2]), int(max_prob_box[3])), (0, 255, 0), 1)

                # for box in boxes:
                #     bounding_boxes.append(box)
                #     print(box)
                #     cls_id = box[-1]

                #     class_list = YOLO_model.model.names
                #     class_name = class_list[cls_id]

                #     cv2.putText(box_image, class_name,
                #             (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                    
                #     cv2.rectangle(box_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
=======
                # max_prob_index = np.argmax(boxes[:, -2])
                # max_prob_box = boxes[max_prob_index]

                #print(max_prob_box)

                # cls_id = max_prob_box[-1]

                # class_list = YOLO_model.model.names
                # class_name = class_list[cls_id]


                # cv2.putText(box_image, class_name,
                #             (int(max_prob_box[0]), int(max_prob_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                    
                # cv2.rectangle(box_image, (int(max_prob_box[0]), int(max_prob_box[1])), (int(max_prob_box[2]), int(max_prob_box[3])), (0, 255, 0), 1)

                

                for box in boxes:
                    bounding_boxes.append(box)
                    #print(box)
                    cls_id = box[-1]

                    class_list = YOLO_model.model.names
                    class_name = class_list[cls_id]

                    cv2.putText(box_image, class_name,
                            (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
                    
                    cv2.rectangle(box_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
>>>>>>> origin/main

                for keypoint in keypoints:
                    for point in keypoint:
                        x, y, prob = point[0], point[1], point[2]  # Extract x, y, and prob from keypoint
                        x, y = int(x), int(y)  # Convert to integers
                        # print(x, y, prob)

                        if prob > 0.3:
                            pose_keypoints_2d.append([x, y])
                            cv2.circle(box_image, (x, y), 2, (0, 0, 255), -1) 
<<<<<<< HEAD
=======
            
            bounding_boxes = np.asarray(bounding_boxes)


            dataset = ViTDetDataset(model.cfg, color_frame, bounding_boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                final_img = np.concatenate([input_patch, regression_img], axis=1)
        


            
>>>>>>> origin/main

        else:
            assert False, "No detections found by YOLOv8"

            

        #2. YOLO ouputs into HMR2_4dhuman

        #3. HMR2_4dhuman outputs smpl pose, bbox, 3d pose

        #4. cv2 video show 



        #Image presentation section
<<<<<<< HEAD
        output_images = [(color_image, 'Camera Input'), 
                         (box_image, 'Bbox&Pose')
                         ]

        for img, text in output_images:
                    cv2.putText(img, text, (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
        output_image = np.concatenate([img for img, _ in output_images], axis=1)   
        cv2.imshow('Output', output_image)
=======
        # output_images = [(color_image, 'Camera Input'), 
        #                  (box_image, 'Bbox&Pose')
        #                  ]

        # for img, text in output_images:
        #             cv2.putText(img, text, (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
        # output_image = np.concatenate([img for img, _ in output_images], axis=1)   
        cv2.imshow('Output', final_img)
>>>>>>> origin/main


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








