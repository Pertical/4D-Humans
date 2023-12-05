from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import time

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from hmr2.models import load_pose_lift
from hmr2.utils.render_openpose import render_openpose

from hmr2.utils.geometry import aa_to_rotmat, perspective_projection, rotmat_to_angle_axis

import shutil

import json
import copy
import glob

# LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
LIGHT_RED = (0.86666667, 0.07843137, 0.23529412)

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    # parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_2DJointAsInput_2023-11-26_09-53-45/checkpoints/last.ckpt', help='Path to pretrained model checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_fcresnet_Nov28/checkpoints/last.ckpt', help='Path to pretrained model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_fc_Nov28/checkpoints/last.ckpt', help='Path to pretrained model checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='logs/train/runs/pose_lift_fc_Nov28/checkpoints/epoch=25-step=470000.ckpt', help='Path to pretrained model checkpoint')
   
    # parser.add_argument('--img_folder', type=str, default='example_data/test_data/in-the-wild', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='demo_out/in-the-wild', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='demo_out/images', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/ck', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/ck_mesh_2dInput_fc_ep25', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/mbg', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/mbg_mesh_2dInput', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/fb', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/fb_mesh_2dInput', help='Output folder to save rendered results')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/fb_mesh_2dInput_fc_ep25', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/yoga', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/yoga_mesh_2dInput1', help='Output folder to save rendered results')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/yoga_mesh_2dInput_fcresnet', help='Output folder to save rendered results')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/yoga_mesh_2dInput_fc', help='Output folder to save rendered results')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/yoga_mesh_2dInput_fc_ep24', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/cxk', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/cxk_mesh_2dInput1', help='Output folder to save rendered results')

    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/c01', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/c01_mesh_2dInput1', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/talk3', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/talk3_mesh_2dInput_fc', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/talk_J', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/talk_J_mesh_2dInput_fc', help='Output folder to save rendered results')
    
    # parser.add_argument('--img_folder', type=str, default='example_data/test_videos/km3_3', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='example_data/test_videos/km3_3_mesh_2dInput_fc', help='Output folder to save rendered results')
    
    parser.add_argument('--img_folder', type=str, default='example_data/test_videos/hb', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='example_data/test_videos/hb_mesh_2dInput_fc', help='Output folder to save rendered results')
    
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=True, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=True, help='If set, render top view also')
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

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    image_path_list = glob.glob(args.img_folder + '/*.jpg') + glob.glob(args.img_folder + '/*.png')
    image_path_list = sorted(image_path_list)
    # image_path_list = image_path_list[420:720]
    # image_path_list = image_path_list[3500:4000] # J
    # image_path_list = image_path_list[100:500] # Y
    # image_path_list = image_path_list[400:] # km3_3
    # image_path_list = image_path_list[410:] # hb
    image_path_list = image_path_list[545:] # hb
    prev_output_path = ""
    for img_path in image_path_list:

        # img_path = image_path_list[177]
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        # dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        ## or using media-pipe to predict 2d-keypoint directly
        
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        
        all_verts = []
        all_cam_t = []
        for batch in dataloader:
            if len(batch['input_keypoints_2d'].shape) == 1:
                src_path = prev_output_path
                dst_path = output_path = os.path.join(args.out_folder, f'{img_fn}_full_2DInput.png')
                shutil.copy(src_path, dst_path)
                print("continue!!!!!", dst_path)
                continue # No person in input image

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
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_RED,
                                        scene_bg_color=(1, 1, 1),
                                        )

                # final_img = np.concatenate([input_patch, regression_img], axis=1)
                input_keypoints_2d = batch['input_keypoints_2d'][n]
                input_keypoints_2d[:, :-1] = input_patch.shape[1] * (input_keypoints_2d[:, :-1] + 0.5)
                input_keypoints_2d = input_keypoints_2d.cpu().numpy().astype(np.float32)

                input_keypoints_2d_img = render_openpose(255*input_patch.copy(), input_keypoints_2d)
                final_img = np.concatenate([255*input_patch, input_keypoints_2d_img, 255*regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_RED,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([final_img, 255*side_img], axis=1)

                if args.top_view:
                    top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_RED,
                                            scene_bg_color=(1, 1, 1),
                                            top_view=True)
                    final_img = np.concatenate([final_img, 255*top_img], axis=1)

                output_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}_2DInput.png')
                cv2.imwrite(output_path, final_img[:, :, ::-1])
                print(output_path)

                # output_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}_2DInput_croped.png')
                # cv2.imwrite(output_path, 255*input_patch.copy()[:, :, ::-1])
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

                
                # Save pose and betas parameters
                # out.keys:  dict_keys(['pred_cam', 'pred_smpl_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d'])
                pred_smpl_params = out["pred_smpl_params"]
                global_orient = pred_smpl_params["global_orient"][n]
                body_pose = pred_smpl_params["body_pose"][n]
                betas = pred_smpl_params["betas"][n]
                pose = torch.cat([global_orient, body_pose], dim=0)
                pose = rotmat_to_angle_axis(pose)
                pose = pose.view(-1).cpu().numpy().astype(np.float32)
                betas = betas.cpu().numpy().astype(np.float32)
                
                out_copy = copy.deepcopy(out)
                for k, v in out.items():
                    if k == 'pred_smpl_params':
                        for k_s, v_s in v.items():
                            out_copy[k][k_s] = v_s.cpu().numpy().astype(np.float32).tolist()
                            
                    if torch.is_tensor(v):
                        out_copy[k] = v.cpu().numpy().astype(np.float32).tolist()
                        
                out_copy["pose"] = pose.tolist()
                out_copy["betas"] = betas.tolist()

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
            print(output_path)
            prev_output_path = output_path
            print(output_path)

            # ## project 3d on the original_image
            # pred_keypoints_3d = out["pred_keypoints_3d"]
            # # pred_keypoints_3d = out["pred_vertices"]
            # pred_cam_t = torch.tensor(all_cam_t).float().to(pred_keypoints_3d.device)
            # full_image_focallen = torch.tensor([[scaled_focal_length, scaled_focal_length]]).float().to(pred_keypoints_3d.device)
            # camera_center = torch.tensor(img_size * 0.5).float().to(pred_keypoints_3d.device)
            # proj_keypoints_2d = perspective_projection(pred_keypoints_3d,
            #                                 translation=pred_cam_t,
            #                                 focal_length=full_image_focallen,
            #                                 camera_center=camera_center
            #                                 )
            
            # ## draw 2d joints
            # proj_keypoints_2d_np = proj_keypoints_2d.cpu().numpy().astype(np.float32)
            # for i in range(proj_keypoints_2d_np.shape[1]):
            #     x, y = proj_keypoints_2d_np[0, i, 0], proj_keypoints_2d_np[0, i, 1]
            #     cv2.circle(input_img_overlay, tuple((int(x), int(y))), 1, (1, 0, 0), -1)

            # output_path = os.path.join(args.out_folder, f'{img_fn}_all_2djoint.jpg')
            # cv2.imwrite(output_path, 255*input_img_overlay[:, :, ::-1])
            # print(output_path)
            # print("-" * 50)

            # verts_z = pred_keypoints_3d.cpu().numpy().astype(np.float32)[:, :, 2]
            # out_copy["verts2d"] = proj_keypoints_2d_np.tolist()
            # out_copy["verts_z"] = verts_z.tolist()
            # out_copy["full_cam_t"] = np.asarray(all_cam_t).tolist()
            # out_copy["full_verts"] = np.asarray(all_verts).tolist()
            # output_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}_all_param.json')
            # with open(output_path, 'w') as json_file:
            #     json.dump(out_copy, json_file)
            # print(output_path)
            # print(out_copy.keys())
            # # dict_keys(['pred_cam', 'pred_smpl_params', 'pred_cam_t', 'focal_length',
            # #  'pred_keypoints_3d', 'pred_vertices',
            # #  'pred_keypoints_2d', 'pose', 'betas', 'verts2d', 'verts_z', 'full_cam_t', 'full_verts'])
            # print("+" * 50)

if __name__ == '__main__':
    main()
