# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.agent = "our"
_C.tasks = "insert_onto_square_peg,open_drawer,place_wine_at_rack_location,light_bulb_in"
_C.exp_id = "def"
# bs per device, effective bs is scaled by num device
_C.bs = 4
_C.epochs = 100
# number of dataloader workers, >= 0
_C.num_workers = 0
# 'transition_uniform' or 'task_uniform'
_C.sample_distribution_mode = 'transition_uniform'
_C.train_iter = 16 * 10000
_C.use_scheduler = True
# Effective batch across all ranks and gradient-accumulation micro-steps.
# Zero keeps the historical one-update-per-DDP-batch behavior.
_C.global_batch_size = 0
# Optional optimizer-step cap. Zero means epochs * steps_per_epoch.
_C.max_optimizer_steps = 0
_C.gradient_checkpointing = False
_C.efficient_paligemma_forward = False
_C.gpu_paligemma_preprocessing = False
_C.flash_attention_2 = False
# Number of leading Gemma decoder layers whose weights are frozen.
_C.freeze_gemma_prefix_layers = 0
# Freeze weights only; the projector always remains in the forward path.
_C.freeze_multimodal_projector = False
_C.seed = 0
_C.checkpoint_every_epochs = 10
# Optional rank-0 training sample visualization. Images are collected only on
# matching optimizer steps, so the disabled default has no training overhead.
_C.train_visualization = CN()
_C.train_visualization.enabled = False
_C.train_visualization.interval = 500
_C.train_visualization.save_png = True
_C.train_visualization.tensorboard = True
_C.train_visualization.output_dir = "train_visualizations"
# Oracle object experiment: opt in only when loading an augmented replay copy.
_C.use_oracle_objects = False
_C.oracle_max_objects = 32
_C.oracle_num_points = 512
_C.oracle_prior_hidden_channels = 16
_C.oracle_prior_adapter_rank = 0
_C.oracle_prior_multiscale_fusion = False
_C.oracle_relation_gated_adapter = False
# Keep Oracle feature residuals out of rotation/gripper/collision branches.
_C.oracle_adapter_translation_only = False
# arguments present in both peract and rvt
# some of them donot support every possible combination in peract
_C.peract = CN()
_C.peract.lambda_weight_l2 = 1e-6
# lr should be thought on per sample basis
# effective lr is multiplied by bs * num_devices
_C.peract.lr = 2.5e-5
# Zero preserves the historical single-learning-rate optimizer.
_C.peract.gemma_lr = 0.0
_C.peract.gemma_layer_lr_decay = 1.0
_C.peract.optimizer_type =  "adam" # "lamb"
_C.peract.add_rgc_loss = True
_C.peract.num_rotation_classes = 72
_C.peract.transform_augmentation = True
_C.peract.transform_augmentation_xyz = [0.1, 0.1, 0.1]
_C.peract.transform_augmentation_rpy = [0.0, 0.0, 20.0]

# arguments present in only rvt and not peract
_C.rvt = CN()
_C.rvt.gt_hm_sigma = 1.5
_C.rvt.img_aug = 0.1
_C.rvt.place_with_mean = True
_C.rvt.move_pc_in_bound = True
# Optional privileged O2 prior. Defaults preserve the historical forward path.
_C.rvt.oracle_prior_mode = 'none'
_C.rvt.oracle_prior_sigma = 2.0
# Used only by the legacy single-prior path.
_C.rvt.oracle_prior_active_role = 'auto'
_C.rvt.oracle_prior_strict = False
_C.rvt.oracle_prior_relation = False
_C.rvt.oracle_log_base_loss = False
# Normalize translation loss over complete Oracle relation samples only.
_C.rvt.oracle_valid_only_loss = False

# arguments present in peract official
_C.peract_official = CN()
_C.peract_official.cfg_path = "configs/peract_official_config.yaml"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
