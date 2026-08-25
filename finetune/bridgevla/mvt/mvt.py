'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.mvt_single import MVT as MVTSingle
from bridgevla.mvt.config import get_cfg_defaults
from bridgevla.models.oracle_prior import (
    OraclePriorFeatureAdapter,
    OraclePriorFusion,
    rasterize_instance_points,
)

class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        img_feat_dim,
        feat_dim,
        im_channels,
        activation,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        stage_two,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        img_aug_2,
        renderer_device,
        load_pretrain=False,
        pretrain_path=None,
        flash_attention_2=False,
        oracle_prior_fusion=False,
        oracle_prior_hidden_channels=16,
        oracle_prior_adapter_rank=0,
        oracle_prior_multiscale_fusion=False,
        oracle_prior_relation=False,
    ):
        super().__init__()
        if oracle_prior_adapter_rank < 0:
            raise ValueError('oracle_prior_adapter_rank must be >= 0')

        from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer

        global BoxRenderer

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]
        del args["oracle_prior_fusion"]
        del args["oracle_prior_hidden_channels"]

        del args['oracle_prior_adapter_rank']
        del args['oracle_prior_multiscale_fusion']
        del args['oracle_prior_relation']

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2
        self.oracle_prior_relation = bool(oracle_prior_relation)
        oracle_prior_channels = 2 if oracle_prior_relation else 1
        self.oracle_prior_fusion1 = (
            OraclePriorFusion(
                oracle_prior_hidden_channels,
                multiscale=oracle_prior_multiscale_fusion,
                prior_channels=oracle_prior_channels,
            )
            if oracle_prior_fusion else None
        )
        self.oracle_prior_fusion2 = (
            OraclePriorFusion(
                oracle_prior_hidden_channels,
                multiscale=oracle_prior_multiscale_fusion,
                prior_channels=oracle_prior_channels,
            )
            if oracle_prior_fusion and stage_two else None
        )

        # for verifying the input
        self.feat_ver = feat_ver
        self.img_feat_dim = img_feat_dim

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.img_size = img_size
        self.mvt1 = MVTSingle(
            **args,
            renderer=self.renderer,
        )  # we have merged mvt1 and mvt2
        use_adapter = oracle_prior_fusion and oracle_prior_adapter_rank > 0
        self.oracle_prior_feature_adapter1 = (
            OraclePriorFeatureAdapter(
                self.mvt1.vlm_dim, oracle_prior_adapter_rank,
                prior_channels=oracle_prior_channels,
            )
            if use_adapter else None
        )
        self.oracle_prior_feature_adapter2 = (
            OraclePriorFeatureAdapter(
                self.mvt1.vlm_dim, oracle_prior_adapter_rank,
                prior_channels=oracle_prior_channels,
            )
            if use_adapter and stage_two else None
        )



    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            assert out is None
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

        return out



    def get_wpt(self, out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            wpt = self.mvt1.get_wpt(
                out, dyn_cam_info, y_q,
            )
        else:
            assert self.stage_two
            wpt = self.mvt1.get_wpt(
                out["mvt2"], dyn_cam_info, y_q
            )
            wpt = out["rev_trans"](wpt)

        return wpt

    def _build_oracle_instance_prior(
        self, points, valid, first_stage, full_out,
        sigma,
    ):
        if points is None:
            return None
        if self.oracle_prior_relation and points.ndim != 4:
            raise ValueError(
                'relation mode requires oracle prior points [B,2,P,3]'
            )
        if not self.oracle_prior_relation and points.ndim != 3:
            raise ValueError(
                'single-prior mode requires oracle prior points [B,P,3]'
            )
        relation_shape = None
        if points.ndim == 4:
            batch_size, role_count, num_points, _ = points.shape
            relation_shape = (batch_size, role_count, num_points)
            points = points.reshape(batch_size, role_count * num_points, 3)
        elif points.ndim != 3:
            raise ValueError(
                'oracle prior points must have shape [B,P,3] or [B,R,P,3]'
            )
        projected = self.get_pt_loc_on_img(
            points, mvt1_or_mvt2=first_stage, dyn_cam_info=None,
            out=None if first_stage else full_out,
        )
        if relation_shape is not None:
            batch_size, role_count, num_points = relation_shape
            projected = projected.reshape(
                batch_size, role_count, num_points,
                projected.shape[-2], 2,
            )
            role_priors = [
                rasterize_instance_points(
                    projected[:, role], valid[:, role],
                    (self.img_size, self.img_size), sigma,
                )
                for role in range(role_count)
            ]
            return torch.stack(role_priors, dim=2)
        return rasterize_instance_points(
            projected, valid, (self.img_size, self.img_size), sigma,
        )

    def _apply_oracle_instance_prior(
        self, stage_out, prior, valid, first_stage,
    ):
        fusion = (
            self.oracle_prior_fusion1
            if first_stage else self.oracle_prior_fusion2
        )
        if prior is None or fusion is None:
            return
        raw_logits = stage_out['trans']
        stage_out['trans_raw'] = raw_logits.detach()
        stage_out['trans'] = fusion(raw_logits, prior, valid)
        if prior.ndim == 5:
            stage_out['oracle_target_prior'] = prior[:, :, 0].detach()
            stage_out['oracle_reference_prior'] = prior[:, :, 1].detach()
            stage_out['oracle_instance_prior'] = prior.amax(dim=2).detach()
        else:
            stage_out['oracle_instance_prior'] = prior.detach()



    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2, dyn_cam_info):
        assert isinstance(mvt1_or_mvt2, bool)

        mvt = self.mvt1

        with torch.no_grad():
            # with autocast(enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info

                if mvt.add_corr:
                    if mvt.norm_corr:
                        img = []
                        for _pc, _img_feat, _dyn_cam_info in zip(
                            pc, img_feat, dyn_cam_info_itr
                        ):
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                           
                            img.append(
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                    fix_cam=True,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                ).unsqueeze(0)
                            )
                    else:
                        img = [
                            self.renderer(
                                _pc,
                                torch.cat((_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None)
                                else None,
                            ).unsqueeze(0)
                            for (_pc, _img_feat, _dyn_cam_info) in zip(
                                pc, img_feat, dyn_cam_info_itr
                            )
                        ]
                else:
                    img = [
                        self.renderer(
                            _pc,
                            _img_feat,
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None)
                            else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            pc, img_feat, dyn_cam_info_itr
                        )
                    ]

        img = torch.cat(img, 0)
        
        img = img.permute(0, 1, 4, 2, 3)

        # for visualization purposes
        if mvt.add_corr:
            mvt.img = img[:, :, 3:].clone().detach()
        else:
            mvt.img = img.clone().detach()

        # image augmentation
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )

        return img


    def verify_inp(
        self,
        pc,
        img_feat,
        img_aug,
        wpt_local,
        rot_x_y,
    ):
        bs = len(pc)
        assert bs == len(img_feat)

        if not self.training:
            # no img_aug when not training
            assert img_aug == 0
            # assert rot_x_y is None, f"rot_x_y={rot_x_y}"

        if self.training:
            assert (
                (not self.feat_ver == 1)
                or (not wpt_local is None)
            )

            if self.rot_ver == 0:
                assert rot_x_y is None, f"rot_x_y={rot_x_y}"
            elif self.rot_ver == 1:
                assert rot_x_y.shape == (bs, 2), f"rot_x_y.shape={rot_x_y.shape}"
                assert (rot_x_y >= 0).all() and (
                    rot_x_y < self.num_rot
                ).all(), f"rot_x_y={rot_x_y}"
            else:
                assert False

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim


        if not (wpt_local is None):
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, "Does not support wpt_local of shape {wpt_local.shape}"

        if self.training:
            assert (not self.stage_two) or (not wpt_local is None)

    def forward(
        self,
        pc,
        img_feat,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        oracle_prior_points=None,
        oracle_prior_valid=None,
        oracle_prior_sigma=2.0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param wpt_local: gt location of the wpt in 3D, tensor of shape
            (bs, 3)
        :param rot_x_y: (bs, 2) rotation in x and y direction
        :param language_goal: str (bs,)language instruction
        """
        self.verify_inp(
            pc=pc,
            img_feat=img_feat,
            img_aug=img_aug,
            wpt_local=wpt_local,
            rot_x_y=rot_x_y,
        )
        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for x in img_feat:
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    # values in [-stdv, stdv]
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise
           
            img = self.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=img_aug,
                mvt1_or_mvt2=True,
                dyn_cam_info=None,
            )
        if self.training:
            wpt_local_stage_one = wpt_local  
            wpt_local_stage_one = wpt_local_stage_one.clone().detach()
        else:
            wpt_local_stage_one = wpt_local

        oracle_prior1 = self._build_oracle_instance_prior(
            oracle_prior_points, oracle_prior_valid, True, None,
            oracle_prior_sigma,
        )
        
   
        out = self.mvt1(
            img=img,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            language_goal=language_goal,
            forward_no_feat=True,
            oracle_prior_heatmap=oracle_prior1,
            oracle_prior_valid=oracle_prior_valid,
            oracle_feature_adapter=(
                self.oracle_prior_feature_adapter1
                if oracle_prior1 is not None else None
            ),
            # forward_no_feat=False,
            **kwargs,
        )
        self._apply_oracle_instance_prior(
            out, oracle_prior1, oracle_prior_valid, True,
        )
        out["mvt1_ori_img"]=img.clone().detach()
        def visualize_tensor(tensor, save_path=None):
            """
            Visualize a tensor of shape (3, 3, 224, 224) as three separate images.
            Each image will show the three channels (RGB) of the corresponding view.
            
            Args:
                tensor: torch.Tensor of shape (3, 3, 224, 224)
                save_path: str, path to save the visualization. If None, only display the image.
            """
            # Convert tensor to numpy array
            tensor_np = tensor.detach().cpu().numpy()
            
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Normalize the values to [0, 1] range
            tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
            
            # Plot each view
            for i in range(3):
                # Transpose from (3, 224, 224) to (224, 224, 3) for RGB display
                img = np.transpose(tensor_np[i], (1, 2, 0))
                axes[i].imshow(img)
                axes[i].set_title(f'View {i+1}')
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path is not None:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Image saved to {save_path}")
            
            plt.show()
        # visualize_tensor(img[0,:,3:6], save_path="/PATH_TO_SAVE_DIR/debug.png")
        if self.stage_two:
            with torch.no_grad():
                # adding then noisy location for training
                if self.training:
                    # noise is added so that the wpt_local2 is not exactly at
                    # the center of the pc
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                    )

                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                        )
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
                        )

                else:
                    # bs, 3
                    wpt_local = self.get_wpt(
                        out, y_q=None, mvt1_or_mvt2=True,
                        dyn_cam_info=None,
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local, sca=self.st_sca
                    )
                    # bad name!
                    wpt_local_stage_one_noisy = wpt_local

                    # must pass None to mvt2 while in eval
                    wpt_local2 = None

                img = self.render(
                    pc=pc,
                    img_feat=img_feat,
                    img_aug=img_aug,
                    mvt1_or_mvt2=False,
                    dyn_cam_info=None,
                )
        
            out['wpt_local1'] = wpt_local_stage_one_noisy
            out['rev_trans'] = rev_trans
            oracle_prior2 = self._build_oracle_instance_prior(
                oracle_prior_points, oracle_prior_valid, False, out,
                oracle_prior_sigma,
            )
            out_mvt2 = self.mvt1(
                img=img,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                language_goal=language_goal,
                forward_no_feat=False,
                oracle_prior_heatmap=oracle_prior2,
                oracle_prior_valid=oracle_prior_valid,
                oracle_feature_adapter=(
                    self.oracle_prior_feature_adapter2
                    if oracle_prior2 is not None else None
                ),
                **kwargs,
            )
            self._apply_oracle_instance_prior(
                out_mvt2, oracle_prior2, oracle_prior_valid, False,
            )

            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans 
            out["mvt2"] = out_mvt2
            out["mvt2_ori_img"]=img.clone().detach()

        return out




if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()
