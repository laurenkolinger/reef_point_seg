"""
Default configuration for the Pipeline Orchestrator.
"""

import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# VICARIUS module context. github_repo/ is the parent of scripts/, which is
# REPO_DIR here. github_repo/ also sits next to inprocess/ (where runs live)
# and config/ (where pipeline.yaml + presets live). In the legacy layout the
# pipeline config does not exist (the old orch is config-less), and step-1's
# default input_dir points at `REPO_DIR/input/TCRMP_CVR`. In the VICARIUS
# layout there is no `scripts/input/` tree; the CVR workbooks stay at the
# source in hopper/ per Commandment I (data lives on NAS / origin).
MODULE_DIR = os.path.dirname(REPO_DIR)                    # github_repo/
MODULE_ROOT = os.path.dirname(MODULE_DIR)                 # modules/cvr_clip_segmentation/
DEFAULT_INPROCESS = os.path.join(MODULE_ROOT, "inprocess")
# CVR raw data (839 Excel workbooks) continues to live at the source location.
DEFAULT_CVR_INPUT = "/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_CVR"

# Default ports (orchestrator picks a free one if taken)
ORCHESTRATOR_PORT = 5050
STAGE_PORTS = {
    2: 5055,
    4: 5065,
    5: 5070,
}

# Python interpreters per stage
PYTHON_PATHS = {
    1: os.path.join(REPO_DIR, "TCRMPcvr_makeAllPoints", "env", "bin", "python"),
    2: os.path.join(REPO_DIR, "TCRMPcvr_recodeSpecies", "env", "bin", "python"),
    3: os.path.join(REPO_DIR, "TCRMPcvr_chooseImages", "env", "bin", "python"),
    4: os.path.join(REPO_DIR, "TCRMPclip_routeChosenImages", "env", "bin", "python"),
    5: "/home/bizon/anaconda3/envs/sam3reef/bin/python",
    # Steps 6/7/8 all run ultralytics, so they share the step-6 env.
    6: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "env", "bin", "python"),
    7: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "env", "bin", "python"),
    8: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "env", "bin", "python"),
}

# Entry points per stage
ENTRY_POINTS = {
    1: os.path.join(REPO_DIR, "TCRMPcvr_makeAllPoints", "run.py"),
    2: os.path.join(REPO_DIR, "TCRMPcvr_recodeSpecies", "src", "app.py"),
    3: os.path.join(REPO_DIR, "TCRMPcvr_chooseImages", "src", "select_images.py"),
    4: os.path.join(REPO_DIR, "TCRMPclip_routeChosenImages", "src", "app.py"),
    5: os.path.join(REPO_DIR, "TCRMPclip_segmentImages", "src", "app.py"),
    # Step 6 isn't a python script — it's a shell driver that chains split + train.
    6: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "run_step6.sh"),
    # Steps 7/8 invoke the same env; entry points are python scripts in src/.
    7: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "src", "evaluate_run.py"),
    8: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV", "src", "run_inference.py"),
}

# Working directories per stage
WORKING_DIRS = {
    1: os.path.join(REPO_DIR, "TCRMPcvr_makeAllPoints"),
    2: os.path.join(REPO_DIR, "TCRMPcvr_recodeSpecies"),
    3: os.path.join(REPO_DIR, "TCRMPcvr_chooseImages"),
    4: os.path.join(REPO_DIR, "TCRMPclip_routeChosenImages"),
    5: os.path.join(REPO_DIR, "TCRMPclip_segmentImages"),
    6: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV"),
    7: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV"),
    8: os.path.join(REPO_DIR, "TCRMPtrain_oceankindCV"),
}

# Default config values per step
DEFAULT_STEP_CONFIGS = {
    "1": {
        "input_dir": DEFAULT_CVR_INPUT,
    },
    "2": {
        "port": 5055,
        "remap_log_source": "",
    },
    "3": {
        "target_species": "OFRA, PA, OA, OFAV, AL, MC, AA",
        "target_instances": 1000,
        "min_year": 2014,
        "max_year": 2025,
        "category_filter": "Coral",
        "skip_image_check": False,
    },
    "4": {
        "port": 5065,
        "clip_dir": "/home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip",
        "target_species_only": True,
        "auto_start_sam3": True,
        # Previously only visible on the sub-app startup form (bypassed by
        # orchestrated auto-boot) — exposed in the Step 4 panel now.
        "reference_mode": False,
        "shuffle": False,
        "review_batch_size": "10",   # "5" | "10" | "20" | "all"
    },
    "5": {
        "port": 5070,
        "sam3_device_tracker": "cuda:1",
        "sam3_device_exemplar": "cuda:0",
        "confidence_threshold": 0.5,
        "min_mask_area_px": 500,
        "merge_distance_px": 30,
        "overlap_strategy": "larger_wins",
        "thin_mask_ratio": 0.10,
        "polygon_simplify_epsilon": 0.001,
        "symlink_images": True,
        # Orchestrator auto-configures step 5, so the sub-app's startup form
        # is never visible. Surface the review batch size here so the user
        # can still pick 5 / 10 / 20 / all.
        "review_batch_size": "10",   # "5" | "10" | "20" | "all"
    },
    "6": {
        # Dataset split
        "valid_ratio": 0.2,
        "test_ratio": 0.1,
        "min_samples": 10,
        # Core training
        "model": "yolo11m-seg.pt",
        "epochs": 500,
        "imgsz": 512,
        "batch": -1,            # -1 = auto
        "patience": 50,
        "optimizer": "auto",    # auto | SGD | AdamW | NAdam | RAdam | RMSProp
        "seed": 0,
        "cos_lr": False,
        "close_mosaic": 10,
        # LR schedule
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        # Loss weights
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "label_smoothing": 0.0,
        # Color augmentation — defaults match oceankind_CV's train_segment.py.
        # Underwater coral recommendations (from 2026 ultralytics research):
        #   hsv_h 0.02–0.05, hsv_s 0.6–0.9, hsv_v 0.5–0.6.
        "hsv_h": 0.2,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "bgr": 0.0,
        # Geometric augmentation
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        # Sample-mixing augmentation
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,       # 0.1–0.3 recommended for rare coral classes
        "copy_paste_mode": "flip",
        "erasing": 0.0,
        "auto_augment": "randaugment",
        # Run name (auto-stamped if left blank)
        "run_name": "",
    },
    "7": {
        # Which training run to evaluate — populated from dropdown listing step 6's runs/.
        "run_dir": "",
        # Split to evaluate against: test | val | train (test is the usual choice).
        "split": "test",
        "imgsz": 512,
        # model.val() does NOT accept -1 (unlike model.train()). Always a positive int.
        "batch": 8,
        "conf_threshold": 0.25,
        "iou_threshold": 0.6,
        # How many test-set preview images to include in the PDF.
        "preview_count": 8,
        # Destination for the exported PDF (in addition to the one saved inside
        # the project's step 7 dir). Optional; leave blank to only keep the local copy.
        "pdf_export_dir": "",
    },
    "8": {
        # Source for inference:
        #   dir        — a directory path
        #   unused     — random / systematic sample of TCRMP frames not in step 3's selected_frames.csv
        #   full       — the entire TCRMP_clip image set
        "source_type": "unused",
        "source_dir": "",
        "sample_count": 100,         # used for random/systematic
        "sample_mode": "random",     # random | systematic
        "include_pts_variant": False,  # include _pts.jpg overlay images or just raw
        # Model:
        "run_dir": "",               # picks <run_dir>/weights/best.pt by default
        "conf_threshold": 0.25,
        "iou_threshold": 0.7,
        "imgsz": 512,
        # Rendering:
        "save_overlays": True,
        "save_crops": False,
        "mask_alpha": 0.45,          # 0.0–1.0 fill opacity for mask overlays
        "draw_boxes": False,         # default off — masks only
        "run_name": "",              # user-friendly tag; stamped in footer + out dir
    },
}
