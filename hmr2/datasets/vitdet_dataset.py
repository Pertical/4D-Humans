from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

try:
    import mediapipe as mp
except:
    raise("Please install mediapipe to predict rough segmentation masks")

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from hmr2.utils.render_openpose import render_openpose


import cv2

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        # print(self.img_size, "------------===============---------=")
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)

        # Initialize mediapipe
        self.pose_mp = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        
        results_mediapipe = self.pose_mp.process(img_patch_cv.astype(np.uint8))
        # pose_landmarks', 'pose_world_landmarks', 'segmentation_mask
        if results_mediapipe.pose_landmarks is None:
            item = {
                    'input_keypoints_2d': -1,
                }
            return item
        
        pose_landmarks = results_mediapipe.pose_landmarks.landmark
        # pose_world_landmarks = results_mediapipe.pose_world_landmarks
        # https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb

        # print(pose_landmarks)
        h, w, c = img_patch_cv.shape
        # image_2djoint = np.copy(img_patch_cv)
        keypoints_2d_list = []
        for point in pose_landmarks:
            x = point.x * w
            y = point.y * h
            z = point.visibility
            # cv2.circle(image_2djoint, (int(x), int(y)), 5, (0, 255, 0), -1)
            keypoints_2d_list.append([x, y, z])
        keypoints_2d_np = np.asarray(keypoints_2d_list).astype(np.float32)
        # print(keypoints_2d_np.shape)
        # # print(keypoint_pos)
        # vis_path = "./demo_out/test_mediapipe.jpg"
        # cv2.imwrite(vis_path, image_2djoint[:, :, ::-1])
        # print(vis_path)

       
        keypoints_2d_openpose_order = self.mediapipe_to_openpose(keypoints_2d_np)
        # input_keypoints_2d_img = render_openpose(img_patch_cv.copy(), keypoints_2d_openpose_order)
        # vis_path = f"./demo_out/demo_input_keypoints_2d_img_{idx}.jpg"
        # cv2.imwrite(vis_path, input_keypoints_2d_img[:, :, ::-1])
        # print(vis_path)

        # for n_jt in range(len(keypoints_2d)):
        #     keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
        keypoints_2d_openpose_order[:, :-1] = keypoints_2d_openpose_order[:, :-1] / patch_width - 0.5
        input_keypoints_2d = keypoints_2d_openpose_order

         # set invalid joint as (0, 0, 0)
        invalid_inds = input_keypoints_2d[:, -1] <= 0
        input_keypoints_2d[invalid_inds] = input_keypoints_2d[invalid_inds] * 0
        input_keypoints_2d = input_keypoints_2d.astype(np.float32)

        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'input_keypoints_2d': input_keypoints_2d,
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        # print("kkkkkkkkkkkkk\n", self.center[idx].copy(), "lllllllllllllll]\n")
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        return item

    def mediapipe_to_openpose(self, mediapipe_keypoints):
        keypoints_2d_openpose_order = np.zeros((25, 3))
        keypoints_2d_openpose_order[0] = mediapipe_keypoints[0]
        # keypoints_2d_openpose_order[1, :-1] = (mediapipe_keypoints[11, :-1] + mediapipe_keypoints[12, :-1]) * 0.5
        # keypoints_2d_openpose_order[1, -1] = mediapipe_keypoints[11, -1] 
        keypoints_2d_openpose_order[2] = mediapipe_keypoints[12]
        keypoints_2d_openpose_order[3] = mediapipe_keypoints[14]
        keypoints_2d_openpose_order[4] = mediapipe_keypoints[16]
        keypoints_2d_openpose_order[5] = mediapipe_keypoints[11]
        keypoints_2d_openpose_order[6] = mediapipe_keypoints[13]
        keypoints_2d_openpose_order[7] = mediapipe_keypoints[15]
        # keypoints_2d_openpose_order[8, :-1] = (mediapipe_keypoints[24, :-1] + mediapipe_keypoints[23, :-1]) * 0.5
        # keypoints_2d_openpose_order[8, -1] = mediapipe_keypoints[24, -1] 
        keypoints_2d_openpose_order[9] = mediapipe_keypoints[24]
        keypoints_2d_openpose_order[10] = mediapipe_keypoints[26]
        keypoints_2d_openpose_order[11] = mediapipe_keypoints[28]
        keypoints_2d_openpose_order[12] = mediapipe_keypoints[23]
        keypoints_2d_openpose_order[13] = mediapipe_keypoints[25]
        keypoints_2d_openpose_order[14] = mediapipe_keypoints[27]
        keypoints_2d_openpose_order[15] = mediapipe_keypoints[4]
        keypoints_2d_openpose_order[16] = mediapipe_keypoints[1]
        keypoints_2d_openpose_order[17] = mediapipe_keypoints[8]
        keypoints_2d_openpose_order[18] = mediapipe_keypoints[7]
        # keypoints_2d_openpose_order[19] = mediapipe_keypoints[29]
        # # keypoints_2d_openpose_order[20] = mediapipe_keypoints[29]
        # keypoints_2d_openpose_order[21] = mediapipe_keypoints[31]
        # keypoints_2d_openpose_order[22] = mediapipe_keypoints[30]
        # # keypoints_2d_openpose_order[23] = mediapipe_keypoints[30]
        # keypoints_2d_openpose_order[24] = mediapipe_keypoints[32]

        return keypoints_2d_openpose_order