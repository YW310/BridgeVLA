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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt_single.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.attn import (
    FixedPositionalEncoding,
)
from bridgevla.mvt.raft_utils import ConvexUpSample
from bridgevla.models.oracle_prior import route_oracle_adapter_features
from PIL import Image



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
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
        load_pretrain=False,
        pretrain_path=None,
        flash_attention_2=False,
    ):
        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img
        # Modify it to adapt to vlm. 16**2 is the number of patches in the image
        self.num_pat_img = 16  

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1


        # Hardcoded for vlm
        self.vlm_dim=2048  

        self.up0 = ConvexUpSample(
            in_dim=self.vlm_dim,
            out_dim=1,
            up_ratio=self.img_patch_size,
        )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.vlm_dim
            # Because we will concatenate the max-pooled image tokens and the image tokens corresponding to the waypoint later.
            if self.cvx_up:
                feat_fc_dim += self.vlm_dim
            else:
                feat_fc_dim += self.final_dim
            

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            if self.rot_ver == 0:
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    feat_out_size,
                )
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, feat_out_size_ex_rot
                    )

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from bridgevla.mvt.renderer import select_feat_from_hm

        from transformers import (
            PaliGemmaProcessor,
            PaliGemmaForConditionalGeneration,
        )
        from safetensors import safe_open
        import json

        def load_all_params(checkpoint_dir):
            # Load the index file
            with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
                index = json.load(f)
            
            all_params = {}
            for shard_file in set(index["weight_map"].values()):
                with safe_open(f"{checkpoint_dir}/{shard_file}", framework="pt") as f:
                    for key in f.keys():
                        # Remove the "module." prefix
                        clean_key = key.replace("module.", "")
                        all_params[clean_key] = f.get_tensor(key)
            return all_params


        model_id = "google/paligemma-3b-pt-224"
        model_kwargs = {"torch_dtype": torch.bfloat16}
        if flash_attention_2:
            flash_device = torch.device(renderer_device)
            if flash_device.type != "cuda":
                raise ValueError(
                    "FlashAttention 2 requires a CUDA renderer_device, got "
                    f"{renderer_device!r}."
                )
            model_kwargs["attn_implementation"] = "flash_attention_2"
            # torchrun sets renderer_device to this process's LOCAL_RANK. Load
            # PaliGemma there directly so Transformers validates FA2 against a
            # CUDA-initialized model instead of warning during a CPU load.
            model_kwargs["device_map"] = {"": str(flash_device)}
        if load_pretrain:
            assert pretrain_path is not None

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
            self.processor = PaliGemmaProcessor.from_pretrained(model_id) 
            pretrained_dir=pretrain_path
            print("The pretrained path is:",pretrained_dir)
            all_params = load_all_params(pretrained_dir)

            # Separate the base model parameters (assuming the original model parameter names do not contain "up0")
            base_params = {k: v for k, v in all_params.items() if not k.startswith("up0.")}

            # Separate the custom layer parameters
            custom_params = {k.replace("up0.",""): v for k, v in all_params.items() if k.startswith("up0.")}
            # Load parameters (strict mode)
            missing_keys, unexpected_keys = self.model.load_state_dict(base_params, strict=False)
            print("Missing keys  base:", missing_keys)  # Should be an empty list
            print("Unexpected keys base:", unexpected_keys) # Should be an empty list
            # Load parameters
            missing_keys_up0, unexpected_keys_up0 = self.up0.load_state_dict(custom_params, strict=True)
            print("Missing keys up0:", missing_keys_up0)  # Should be an empty list
            print("Unexpected keys up0 :", unexpected_keys_up0) # Should be an empty list
            import time
            time.sleep(5)

            
        else:

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)   
            print("You are loading original paligemma model!")

        if flash_attention_2:
            attention_backend = self.model.config.text_config._attn_implementation
            if attention_backend != "flash_attention_2":
                raise RuntimeError(
                    "FlashAttention 2 was requested but PaliGemma loaded "
                    f"attention backend {attention_backend!r}."
                )
            print("Enabled PaliGemma FlashAttention 2")

        global select_feat_from_hm
        self.use_efficient_paligemma_forward = False
        self.use_gpu_paligemma_preprocessing = False

    def enable_gradient_checkpointing(self):
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        print('Enabled PaliGemma gradient checkpointing')

    def enable_efficient_paligemma_forward(self):
        self.use_efficient_paligemma_forward = True
        print('Enabled memory-efficient PaliGemma forward')

    def enable_gpu_paligemma_preprocessing(self):
        image_processor = self.processor.image_processor
        required_operations = {
            'resize': image_processor.do_resize,
            'rescale': image_processor.do_rescale,
            'normalize': image_processor.do_normalize,
        }
        disabled = [name for name, enabled in required_operations.items() if not enabled]
        if disabled:
            raise RuntimeError(
                'GPU PaliGemma preprocessing requires the standard resize, '
                'rescale, and normalize operations; disabled: '
                + ', '.join(disabled)
            )
        size = image_processor.size
        if 'height' not in size or 'width' not in size:
            raise RuntimeError(
                'GPU PaliGemma preprocessing requires explicit image height '
                f'and width, got {size!r}.'
            )
        self.use_gpu_paligemma_preprocessing = True
        print('Enabled GPU-native PaliGemma preprocessing')

    def _forward_action_heads(
        self, feat, rot_x_y, batch_size,
    ):
        if self.rot_ver == 0:
            return {'feat': self.feat_fc(feat)}
        if self.rot_ver != 1:
            raise ValueError(f'Unsupported rot_ver: {self.rot_ver}')

        feat_ex_rot = self.feat_fc_ex_rot(feat)
        action_batch_norm = self.feat_fc_init_bn
        batch_norm_frozen = not any(
            parameter.requires_grad
            for parameter in action_batch_norm.parameters()
        )
        if self.training and batch_norm_frozen:
            feat_rot = F.batch_norm(
                feat,
                action_batch_norm.running_mean,
                action_batch_norm.running_var,
                action_batch_norm.weight,
                action_batch_norm.bias,
                training=False,
                momentum=action_batch_norm.momentum,
                eps=action_batch_norm.eps,
            )
        else:
            feat_rot = action_batch_norm(feat)
        feat_x = self.feat_fc_x(feat_rot)
        rot_x = (
            rot_x_y[..., 0].view(batch_size, 1)
            if self.training else feat_x.argmax(dim=1, keepdim=True)
        )
        feat_y = self.feat_fc_y(feat_rot + self.feat_fc_pe(rot_x))
        rot_y = (
            rot_x_y[..., 1].view(batch_size, 1)
            if self.training else feat_y.argmax(dim=1, keepdim=True)
        )
        feat_z = self.feat_fc_z(feat_rot + self.feat_fc_pe(rot_x)
                                + self.feat_fc_pe(rot_y))
        return {
            'feat_ex_rot': feat_ex_rot,
            'feat_x': feat_x,
            'feat_y': feat_y,
            'feat_z': feat_z,
        }

    def _forward_base_action_heads(self, feat, rot_x_y, batch_size):
        '''Run the diagnostic base branch without changing BatchNorm state.'''
        batch_norm_state = None
        if self.rot_ver == 1 and self.training:
            batch_norm = self.feat_fc_init_bn
            batch_norm_state = {
                name: value.detach().clone()
                for name, value in batch_norm.named_buffers(recurse=False)
                if value is not None
            }
        try:
            with torch.no_grad():
                return self._forward_action_heads(
                    feat, rot_x_y, batch_size,
                )
        finally:
            if batch_norm_state is not None:
                with torch.no_grad():
                    for name, value in batch_norm_state.items():
                        getattr(batch_norm, name).copy_(value)

    @staticmethod
    def _build_paligemma_input_strings(
        prompts, image_token, image_seq_length, num_images, bos_token
    ):
        image_prefix = image_token * (image_seq_length * num_images)
        return [f'{image_prefix}{bos_token}{prompt}\n' for prompt in prompts]

    def _prepare_paligemma_inputs_gpu(self, prompts, images):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                'Expected RGB images shaped [batch, views, 3, height, width], '
                f'got {tuple(images.shape)}.'
            )
        batch_size, num_images = images.shape[:2]
        if len(prompts) != batch_size:
            raise ValueError(
                f'Received {len(prompts)} prompts for batch size {batch_size}.'
            )

        image_processor = self.processor.image_processor
        target_size = (
            image_processor.size['height'],
            image_processor.size['width'],
        )
        # Match the legacy Tensor -> NumPy uint8 -> PIL conversion on GPU.
        pixel_values = (images.flatten(0, 1) * 255).to(torch.uint8)
        pixel_values = F.interpolate(
            pixel_values.float(),
            size=target_size,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        # PIL resize returns uint8 pixels. Round before applying the processor's
        # rescale and normalize constants to retain the same value domain.
        pixel_values = pixel_values.round().clamp_(0, 255)
        pixel_values.mul_(float(image_processor.rescale_factor))
        mean = torch.as_tensor(
            image_processor.image_mean,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1, 1)
        std = torch.as_tensor(
            image_processor.image_std,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1, 1)
        pixel_values.sub_(mean).div_(std)
        pixel_values = pixel_values.to(dtype=self.model.dtype)

        tokenizer = self.processor.tokenizer
        input_strings = self._build_paligemma_input_strings(
            prompts=prompts,
            image_token='<image>',
            image_seq_length=self.processor.image_seq_length,
            num_images=num_images,
            bos_token=tokenizer.bos_token,
        )
        text_inputs = tokenizer(
            input_strings,
            padding='longest',
            return_token_type_ids=False,
            return_tensors='pt',
        )
        model_device = self.model.device
        model_inputs = {
            key: value.to(model_device) for key, value in text_inputs.items()
        }
        model_inputs['pixel_values'] = pixel_values.to(model_device)
        return model_inputs

    def _forward_efficient_paligemma(self, model_inputs):
        # PaliGemma 4.51.3 performs multimodal token merging in the conditional
        # generation wrapper, so keep that path but capture only the final
        # normalized hidden state. Limiting logits to one token avoids the
        # otherwise unused [batch, sequence, vocabulary] allocation, while not
        # requesting every hidden layer preserves activation checkpointing.
        captured_hidden_states = []

        def capture_final_hidden_state(module, inputs, output):
            captured_hidden_states.append(output)

        hook = self.model.language_model.model.norm.register_forward_hook(
            capture_final_hidden_state
        )
        try:
            self.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
                return_dict=True,
            )
        finally:
            hook.remove()
        if len(captured_hidden_states) != 1:
            raise RuntimeError(
                'Expected one final PaliGemma hidden state, got '
                f'{len(captured_hidden_states)}'
            )
        return captured_hidden_states[0]

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        Transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    @staticmethod
    def trans_cuda_tensor_2_PIL(cuda_tensor):
        # Default c,h,w, and 0,1
        # 1. Move the tensor from GPU to CPU
        tensor_cpu = cuda_tensor.cpu()

        # 2. Convert to a numpy array and adjust the dimension order [3, 224, 224] -> [224, 224, 3]
        image = tensor_cpu.permute(1, 2, 0).numpy()

        # 3. Convert the values from [0, 1] to integers in [0, 255] and cast to uint8 type
        image = (image * 255).astype('uint8')

        # 4. Create a PIL image object
        pil_image = Image.fromarray(image)

        # 5. Convert to RGB format (ensure the image is RGB)
        pil_image_rgb = pil_image.convert("RGB")
        return pil_image_rgb

    def forward(
        self,
        img,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        forward_no_feat=False,
        oracle_prior_heatmap=None,
        oracle_prior_valid=None,
        oracle_feature_adapter=None,
        oracle_relation_points=None,
        oracle_adapter_translation_only=False,
        oracle_compute_base=False,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param rot_x_y: (bs, 2)
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        assert num_img == self.num_img
        assert h == w == self.img_size
        # only use rgb part
        # print("input image feature shape:",img.shape)
        img = img[:,:, 3:6, :, :] # bs,3,3,224,224


        prompts =[ text[0][0] for text in language_goal]# ["text1","text2"...]
        if self.use_gpu_paligemma_preprocessing:
            model_inputs = self._prepare_paligemma_inputs_gpu(prompts, img)
        else:
            images = [
                [MVT.trans_cuda_tensor_2_PIL(example) for example in examples]
                for examples in img
            ]
            assert len(prompts) == len(images)
            model_inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors='pt',
                padding='longest',
            )
            model_inputs = model_inputs.to(self.model.dtype).to(self.model.device)
        if self.use_efficient_paligemma_forward:
            x = self._forward_efficient_paligemma(model_inputs)
        else:
            outputs = self.model(**model_inputs, output_hidden_states=True)
            x = outputs.hidden_states[-1]


        # get image tokens
        image_tokens= []

        # Process every batch
        for i in range(bs):
            # Get the ids and output of the current batch
            current_ids = model_inputs["attention_mask"][i]
            current_output = x[i]
            
            # Extract tokens corresponding to non-zero ids
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]  # Find the indices of non-zero ids
            non_zero_output = current_output[non_zero_indices]  # Extract the token outputs corresponding to these non-zero ids
            
            # Take the first 256 tokens (if the number of non-zero tokens is greater than 256, take the first 256)
            assert non_zero_output.shape[0] > 256*self.num_img
            non_zero_output = non_zero_output[:256*self.num_img]
            
            # Add the processed output to the new output list
            image_tokens.append(non_zero_output)

        # concat all the output
        image_tokens = torch.stack(image_tokens)
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=self.num_img, h1=self.num_pat_img, h2=self.num_pat_img) 
        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32)
        base_action_features = x
        trans_base = None
        translation_features = x
        if oracle_feature_adapter is not None:
            if oracle_prior_heatmap is None or oracle_prior_valid is None:
                raise ValueError('Oracle feature adapter requires prior and valid')
            if oracle_compute_base:
                with torch.no_grad():
                    trans_base = self.up0(x).view(
                        bs, self.num_img, h, w,
                    )
            translation_features = oracle_feature_adapter(
                x, oracle_prior_heatmap, oracle_prior_valid,
                oracle_relation_points,
            )
            translation_features, x = route_oracle_adapter_features(
                x,
                translation_features,
                oracle_adapter_translation_only,
            )
        
        trans = self.up0(translation_features)
        trans = trans.view(bs, self.num_img, h, w)


        if not forward_no_feat:

            # get wpt_local while testing
            if not self.training:
                wpt_local = self.get_wpt(
                    out={"trans": trans.clone().detach()},
                    dyn_cam_info=None,
                )

            # projection
            # (bs, 1, num_img, 2)
            wpt_img = self.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                dyn_cam_info=None,
            )
            wpt_img = wpt_img.reshape(bs * self.num_img, 2)

            # add noise to wpt image while training
            if self.training:
                wpt_img = mvt_utils.add_uni_noi(
                    wpt_img, self.wpt_img_aug * self.img_size
                )
                wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

            _wpt_img = wpt_img / self.img_patch_size
            _u = x
            assert (
                0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
            ), print(_wpt_img, x.shape)

            _wpt_img = _wpt_img.unsqueeze(1)
            base_action_out = None
            if oracle_compute_base and oracle_feature_adapter is not None:
                with torch.no_grad():
                    base_local_feat = select_feat_from_hm(
                        _wpt_img, base_action_features,
                    )[0].view(bs, -1)
                    base_action_input = torch.cat(
                        (feat[0], base_local_feat), dim=-1,
                    )
                    base_action_out = self._forward_base_action_heads(
                        base_action_input, rot_x_y, bs,
                    )
            _feat = select_feat_from_hm(_wpt_img, _u)[0]
            _feat = _feat.view(bs, -1)
            feat.append(_feat)
            feat = torch.cat(feat, dim=-1)

            out = self._forward_action_heads(
                feat, rot_x_y, bs,
            )
            if base_action_out is not None:
                out.update({
                    f'{name}_base': value.detach()
                    for name, value in base_action_out.items()
                })
        
        else:
            out = {}

        out.update({"trans": trans})

        if trans_base is not None:
            out['trans_base'] = trans_base
        return out





    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt


    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
