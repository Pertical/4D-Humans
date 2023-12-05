from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from hmr2.models import load_pose_lift
from hmr2.utils.render_openpose import render_openpose
from hmr2.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

from shutil import copy

try:
    import mediapipe as mp
except:
    raise("Please install mediapipe to predict rough segmentation masks")

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import glob

# LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
LIGHT_RED = (0.86666667, 0.07843137, 0.23529412)

def get_bbox_from_2djoints(keypoints_2d_np, scale=1.0):
    x_coordinates = keypoints_2d_np[:, 0]
    y_coordinates = keypoints_2d_np[:, 1]

    min_x, min_y = min(x_coordinates), min(y_coordinates)
    max_x, max_y = max(x_coordinates), max(y_coordinates)

    width = max_x - min_x
    height = max_y - min_y

    # Apply the scaling factor
    scaled_width = width * scale
    scaled_height = height * scale

    # Calculate the center of the original bounding box
    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Adjust the bounding box to maintain the center and apply scaling
    new_min_x = int(center_x - scaled_width / 2)
    new_min_y = int(center_y - scaled_height / 2)

    bbox = (new_min_x, new_min_y, int(scaled_width), int(scaled_height))

    return bbox


def convert_to_square_bbox(bbox):
    x, y, w, h = bbox

    # Find the center of the original bounding box
    center_x = x + w / 2
    center_y = y + h / 2

    # Determine the size of the square bounding box (maximum of width and height)
    size = max(w, h)

    # Calculate the new coordinates for the square bounding box
    new_x = int(center_x - size / 2)
    new_y = int(center_y - size / 2)

    square_bbox = (new_x, new_y, int(size), int(size))

    return square_bbox


def get_center_and_size_from_bbox(bbox):
    x, y, width, height = bbox

    # Calculate the center coordinates
    center_x = x + width / 2
    center_y = y + height / 2

    # Size of the bounding box
    box_size = max(width, height)

    # Center coordinates of the bounding box
    box_center = (center_x, center_y)

    return box_center, box_size



def mediapipe_to_openpose(mediapipe_keypoints):
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


def crop_image(image, bbox):
    x, y, width, height = bbox
    x = max(0, x)
    y = max(0, y)
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    # parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_2DJointAsInput_2023-11-26_09-53-45/checkpoints/last.ckpt', help='Path to pretrained model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_fc_Nov28/checkpoints/last.ckpt', help='Path to pretrained model checkpoint')

    # parser.add_argument('--img_folder', type=str, default='example_data/test_data/in-the-wild', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='demo_out/in-the-wild', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='demo_out/images', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/ck', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/ck_mesh_2dInput', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/mbg', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/mbg_mesh_2dInput', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/fb', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/fb_mesh_2dInput', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/cxk', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/cxk_mesh_2dInput', help='Output folder to save rendered results')
  
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/yoga', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/yoga_mesh_2dInput', help='Output folder to save rendered results')

    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/talk_Y', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/talk_Y_mesh_2dInput_fc_2', help='Output folder to save rendered results')
    
    parser.add_argument('--img_folder', type=str, default='example_data/test_videos/km3_3', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='example_data/test_videos/km3_3_mesh_2dInput_fc_2', help='Output folder to save rendered results')
    

    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')

    args = parser.parse_args()

    # # Download and load checkpoints
    # download_models(CACHE_DIR_4DHUMANS)
    # model, model_cfg = load_hmr2(args.checkpoint)

    model, model_cfg = load_pose_lift(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Initialize mediapipe
    pose_mp = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    image_path_list = glob.glob(args.img_folder + '/*.jpg') + glob.glob(args.img_folder + '/*.png')
    image_path_list = sorted(image_path_list)
    # image_path_list = image_path_list[3500:4000] # J
    # image_path_list = image_path_list[100:400] # Y
    # image_path_list = image_path_list[400:]
    image_path_list = image_path_list[500:]

    prev_output_path = ""
    for img_path in image_path_list:
        img_cv2 = cv2.imread(str(img_path))
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        print(img_path)
        start_time = time.time()
        ## using media-pipe to predict 2d-keypoint directly
        # https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
        results_mediapipe = pose_mp.process(img_cv2.astype(np.uint8))
        if not results_mediapipe.pose_landmarks:
            src_path = prev_output_path
            dst_path = output_path = os.path.join(args.out_folder, f'{img_fn}_full_2DInput.png')
            copy(src_path, dst_path)
            print("continue!!!!!", dst_path)
            continue # No person in input image
        
        pose_landmarks = results_mediapipe.pose_landmarks.landmark
        mask = (results_mediapipe.segmentation_mask>0.5)*255
        print("mediapipe time: ", time.time() - start_time)

        h, w, c = img_cv2.shape
        keypoints_2d_list = []
        for point in pose_landmarks:
            x = point.x * w
            y = point.y * h
            z = point.visibility
            # cv2.circle(img_cv2, (int(x), int(y)), 5, (0, 255, 0), -1)
            keypoints_2d_list.append([x, y, z])
        keypoints_2d_np = np.asarray(keypoints_2d_list).astype(np.float32)

        # bbox = get_bbox_from_2djoints(keypoints_2d_np, scale=1.1)
        bbox = cv2.boundingRect(np.uint8(mask))
        bbox = convert_to_square_bbox(bbox)
        
        # vis_path = "./demo_out/mask.jpg"
        # cv2.imwrite(vis_path, mask)
        # print(vis_path)

        x, y, width, height = bbox
        x = max(0, x)
        y = max(0, y)
        bbox = (x, y, width, height)
        box_center, box_size = get_center_and_size_from_bbox(bbox)

        # cv2.rectangle(img_cv2, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        # vis_path = "./demo_out/test_mediapipe.jpg"
        # cv2.imwrite(vis_path, img_cv2)
        # print(vis_path)

        left_top_x = box_center[0] - box_size * 0.5
        left_top_y = box_center[1] - box_size * 0.5

        keypoints_2d_np[:, 0] = keypoints_2d_np[:, 0] - left_top_x
        keypoints_2d_np[:, 1] = keypoints_2d_np[:, 1] - left_top_y

        keypoints_2d_openpose_order = mediapipe_to_openpose(keypoints_2d_np)
        cropped_image = crop_image(img_cv2, bbox)
        cropped_image = cropped_image[:, :, ::-1]

        # input_keypoints_2d_img = render_openpose(cropped_image.copy(), keypoints_2d_openpose_order)
        # vis_path = f"./demo_out/test.jpg"
        # cv2.imwrite(vis_path, input_keypoints_2d_img[:, :, ::-1])
        # print(vis_path)

        keypoints_2d_openpose_order[:, :-1] = keypoints_2d_openpose_order[:, :-1] / box_size - 0.5
        # keypoints_2d_openpose_order[:, :-1] = keypoints_2d_openpose_order[:, :-1] / 256 - 0.5
        input_keypoints_2d = keypoints_2d_openpose_order

         # set invalid joint as (0, 0, 0)
        invalid_inds = input_keypoints_2d[:, -1] <= 0
        input_keypoints_2d[invalid_inds] = input_keypoints_2d[invalid_inds] * 0
        input_keypoints_2d = input_keypoints_2d.astype(np.float32)

        cropped_image = cv2.resize(cropped_image, (256, 256)) # match the model input is 256p
        img_patch = convert_cvimg_to_tensor(cropped_image)
        # apply normalization
        for n_c in range(min(cropped_image.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - DEFAULT_MEAN[n_c]) / DEFAULT_STD[n_c]
        
        ## need to check
        # img_patch = img_patch[::-1, :, :]

        batch = {
            'input_keypoints_2d': torch.from_numpy(input_keypoints_2d[None, ...]),
            'img': torch.from_numpy(img_patch.astype(np.float32)[None, ...]),
            'personid': torch.from_numpy(np.array([0]).astype(np.float32)[None, ...]),
        }
        batch['box_center'] = torch.from_numpy(np.array(box_center).astype(np.float32)[None, ...])
        batch['box_size'] = torch.from_numpy(np.array(box_size).astype(np.float32)[None, ...])
        batch['img_size'] = torch.from_numpy(1.0 * np.array([img_cv2.shape[1], img_cv2.shape[0]])[None, ...])

        all_verts = []
        all_cam_t = []
        
        batch = recursive_to(batch, device)
        with torch.no_grad():

            start_time = time.time()
            out = model(batch)
            print("inference time: ", time.time() - start_time)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['box_center'].shape[0]
        for n in range(batch_size):
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            # regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #                         out['pred_cam_t'][n].detach().cpu().numpy(),
            #                         batch['img'][n],
            #                         mesh_base_color=LIGHT_RED,
            #                         scene_bg_color=(1, 1, 1),
            #                         )

            # if args.side_view:
            #     side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #                             out['pred_cam_t'][n].detach().cpu().numpy(),
            #                             white_img,
            #                             mesh_base_color=LIGHT_RED,
            #                             scene_bg_color=(1, 1, 1),
            #                             side_view=True)
            #     final_img = np.concatenate([final_img, side_img], axis=1)

            # if args.top_view:
            #     top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #                             out['pred_cam_t'][n].detach().cpu().numpy(),
            #                             white_img,
            #                             mesh_base_color=LIGHT_RED,
            #                             scene_bg_color=(1, 1, 1),
            #                             top_view=True)
            #     final_img = np.concatenate([final_img, top_img], axis=1)

            input_keypoints_2d = batch['input_keypoints_2d'][n]
            input_keypoints_2d[:, :-1] = input_patch.shape[1] * (input_keypoints_2d[:, :-1] + 0.5)
            input_keypoints_2d = input_keypoints_2d.cpu().numpy().astype(np.float32)

            # input_keypoints_2d_img = render_openpose(255*input_patch.copy(), input_keypoints_2d)
            # final_img = np.concatenate([255*input_patch, input_keypoints_2d_img, 255*regression_img], axis=1)

            # output_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}_2DInput.png')
            # cv2.imwrite(output_path, final_img[:, :, ::-1])
            # print(output_path)

            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)

            # # Save all meshes to disk
            # if args.save_mesh:
            #     camera_translation = cam_t.copy()
            #     tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_RED)
            #     tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_RED,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            left_top_x = box_center[n][0] - box_size[n] * 0.5
            left_top_y = box_center[n][1] - box_size[n] * 0.5
            scale_bbox = box_size[n] / 256

            input_keypoints_2d_orig = input_keypoints_2d.copy() * scale_bbox.cpu().numpy().astype(np.float32)
            input_keypoints_2d_orig[:, 0] = input_keypoints_2d_orig[:, 0] + left_top_x.cpu().numpy().astype(np.float32)
            input_keypoints_2d_orig[:, 1] = input_keypoints_2d_orig[:, 1] + left_top_y.cpu().numpy().astype(np.float32)
            input_keypoints_2d_img = render_openpose(img_cv2.copy(), input_keypoints_2d_orig)
            # final_img = np.concatenate([img_cv2, input_keypoints_2d_img, 255*input_img_overlay[:, :, ::-1]], axis=1)
            final_img = np.concatenate([input_keypoints_2d_img, 255*input_img_overlay[:, :, ::-1]], axis=1)

            output_path = os.path.join(args.out_folder, f'{img_fn}_full_2DInput.png')
            cv2.imwrite(output_path, final_img)
            prev_output_path = output_path
            print(output_path)

if __name__ == '__main__':
    main()
