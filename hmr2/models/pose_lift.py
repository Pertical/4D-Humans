import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection, convet_25joint_to_17joint
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smpl_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import SMPL

import torch.nn as nn

# from hmr2.models.backbones.fcresnet import FCResNet
from hmr2.models.components.pose_transformer import Transformer, MyUnsqueeze

log = get_pylogger(__name__)

class PoseLift(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup PoseLift model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg

        print("cfg.MODEL.BACKBONE.TYPE: ", cfg.MODEL.BACKBONE.TYPE)
        if cfg.MODEL.BACKBONE.TYPE == 'fc_25joint':
            ## IKnet
            self.joint2d_encode = nn.Sequential(
                    nn.Linear(cfg.MODEL.BACKBONE.input_dim, cfg.MODEL.BACKBONE.output_dim),
                    nn.BatchNorm1d(cfg.MODEL.BACKBONE.output_dim),
                    nn.ReLU(inplace=True)
                )
        elif cfg.MODEL.BACKBONE.TYPE == 'fc':
            ## IKnet
            self.joint2d_encode = nn.Sequential(
                    nn.Linear(cfg.MODEL.BACKBONE.input_dim, cfg.MODEL.BACKBONE.output_dim),
                    nn.BatchNorm1d(cfg.MODEL.BACKBONE.output_dim),
                    nn.ReLU(inplace=True)
                )
        # elif cfg.MODEL.BACKBONE.TYPE == 'fcresnet':
        #     ## ProHMR
        #     self.joint2d_encode = FCResNet(in_channels=cfg.MODEL.BACKBONE.input_dim, 
        #                                    hidden_channels=cfg.MODEL.BACKBONE.hidden_dim, 
        #                                    out_channels=cfg.MODEL.BACKBONE.output_dim)
        elif cfg.MODEL.BACKBONE.TYPE == 'fcTransformer':
            ## transformer encoder
           
            # self.joint2d_encode =nn.Sequential(

            #         Transformer(dim = cfg.MODEL.BACKBONE.input_dim,
            #                     depth = 8,
            #                     heads = 2,
            #                     dim_head = int(cfg.MODEL.BACKBONE.input_dim/2),
            #                     mlp_dim = cfg.MODEL.BACKBONE.hidden_dim,
            #                     dropout = 0.1,
            #                     norm = "layer",
            #                     norm_cond_dim = -1),#B(64),1,34
            #         nn.Flatten(),
            #         nn.Linear(cfg.MODEL.BACKBONE.input_dim, cfg.MODEL.BACKBONE.output_dim),
            #         nn.BatchNorm1d(cfg.MODEL.BACKBONE.output_dim)
            #         # nn.ReLU(inplace=True),
            #     ) 
            self.joint2d_encode =nn.Sequential(
                nn.Linear(cfg.MODEL.BACKBONE.input_dim, cfg.MODEL.BACKBONE.output_dim),
                nn.BatchNorm1d(cfg.MODEL.BACKBONE.output_dim),
                nn.ReLU(inplace=True),
                MyUnsqueeze(dim=1),#64,1,1280
                Transformer(dim = cfg.MODEL.BACKBONE.output_dim,
                            depth = cfg.MODEL.BACKBONE.depth,
                            heads = cfg.MODEL.BACKBONE.heads,
                            dim_head = int(cfg.MODEL.BACKBONE.output_dim/cfg.MODEL.BACKBONE.heads),
                            mlp_dim = cfg.MODEL.BACKBONE.hidden_dim,
                            dropout = cfg.MODEL.BACKBONE.dropout,
                            norm = "layer",
                            norm_cond_dim = -1),#B(64),1,1280
                nn.Flatten(),
                nn.Linear(cfg.MODEL.BACKBONE.output_dim, cfg.MODEL.BACKBONE.output_dim),
                nn.BatchNorm1d(cfg.MODEL.BACKBONE.output_dim)
                # nn.ReLU(inplace=True),
            ) 
                   
        ## backbone from MotionBert todo...
        
        # Create SMPL head
        self.smpl_head = build_smpl_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        # self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1', reduction=cfg.TRAIN.LOSS_REDUCTION)
        # self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1', reduction=cfg.TRAIN.LOSS_REDUCTION)
        # self.smpl_parameter_loss = ParameterLoss(reduction=cfg.TRAIN.LOSS_REDUCTION)

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()


        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        # all_params += list(self.backbone.parameters())
        all_params += list(self.joint2d_encode.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        # x = batch['img']
        # batch_size = x.shape[0]

        # print("x.shape: ", x.shape)
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        # conditioning_feats = self.backbone(x[:,:,:,32:-32])

        # print("keypoints_2d.shape: ", keypoints_2d.shape)               # torch.Size([4, 44, 3])
        # print("conditioning_feats.shape: ", conditioning_feats.shape)   # torch.Size([4, 1280, 16, 12])

         # replace vit backbone with 2d_joint encoder
        # refer to pro_hmr, motionbert, and previous 2d_3d_lifting
        # https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py#L333C25-L333C25
        # https://github.com/nkolot/ProHMR/blob/master/prohmr/models/proskeleton.py#L80
        # https://github.com/nkolot/ProHMR/blob/master/prohmr/models/backbones/fcresnet.py#L52C16-L52C16

        # keypoints_2d = batch['keypoints_2d']      # gt [4, 44, 3]
        keypoints_2d_25joint = batch['input_keypoints_2d']  # selected points [4, 25, 3]
        batch_size = keypoints_2d_25joint.shape[0]

        # if self.cfg.MODEL.BACKBONE.TYPE == 'fc_25joint':
        #      keypoints_2d_17joint = keypoints_2d_25joint
        # else:
        #     keypoints_2d_17joint = convet_25joint_to_17joint(keypoints_2d_25joint)
        # if self.cfg.MODEL.BACKBONE.TYPE == 'fcTransformer':
        #     keypoints_2d_flatten = keypoints_2d_17joint[:, :, :2].reshape(batch_size, -1)
        # else:
        #     keypoints_2d_flatten = keypoints_2d_17joint[:, :, :2].reshape(batch_size, -1)

        # if self.cfg.MODEL.BACKBONE.TYPE == 'fc_25joint':
        #      keypoints_2d_17joint = keypoints_2d_25joint
        # else:
        #     keypoints_2d_17joint = keypoints_2d_25joint
        # if self.cfg.MODEL.BACKBONE.TYPE == 'fcTransformer':
        #     keypoints_2d_flatten = keypoints_2d_17joint[:, :, :2].reshape(batch_size, -1)
        # else:
        #     keypoints_2d_flatten = keypoints_2d_17joint[:, :, :2].reshape(batch_size, -1)

        keypoints_2d_flatten = keypoints_2d_25joint
        


        
        conditioning_feats = self.joint2d_encode(keypoints_2d_flatten) 
        # print("conditioning_feats.shape: ", conditioning_feats.shape)           # [4, 17, 3]
        # print("keypoints_2d.shape: ", keypoints_2d.shape)                           # [4, 1280]
        # print("conditioning_feats.shape: ", conditioning_feats.shape)               # [4, 1280]
        conditioning_feats = conditioning_feats.reshape(batch_size, -1, 1, 1)      
        # print("conditioning_feats.shape: ", conditioning_feats.shape)               # [4, 1280, 1, 1]

        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        input_keypoints_2d = batch['input_keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               input_keypoints_2d[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())

        if self.cfg.TRAIN.LOSS_REDUCTION == 'sum':
            loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        elif self.cfg.TRAIN.LOSS_REDUCTION == 'mean':
            loss_fake = ((disc_fake_out - 0.0) ** 2).mean()

        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        if self.cfg.TRAIN.LOSS_REDUCTION == 'sum':
            loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        elif self.cfg.TRAIN.LOSS_REDUCTION == 'mean':
            loss_real = ((disc_real_out - 1.0) ** 2).mean()

        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            if self.cfg.TRAIN.LOSS_REDUCTION == 'sum':
                loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            elif self.cfg.TRAIN.LOSS_REDUCTION == 'mean':
                loss_adv = ((disc_out - 1.0) ** 2).mean() 
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
