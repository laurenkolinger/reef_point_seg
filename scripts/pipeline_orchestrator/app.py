"""
Flask application for the TCRMP CVR-CLIP Pipeline Orchestrator.
"""

import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

from flask import Flask, render_template, request, jsonify, send_from_directory

import project_manager as pm
import remap_loader
import vicarius_bridge as vic
from stage_runner import StageRunner, find_free_port
from orchestrator_config import (
    REPO_DIR, PYTHON_PATHS, ENTRY_POINTS, WORKING_DIRS, STAGE_PORTS,
    PRESET_DIRS,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
runner = StageRunner()
current_project = None  # dict (project.json contents)

# SAM3 driver state — updated by a background thread per step 5 launch so the
# orchestrator UI can show progress without the user having to open the sub-app.
sam3_status = {
    "phase": "idle",       # idle | launching | waiting_for_sub_app | configuring |
                           # segmenting | review_ready | error
    "processed": 0,
    "total": 0,
    "running": False,
    "error": None,
    "message": "",
    "updated_at": None,
}
sam3_lock = threading.Lock()
sam3_thread = None


def _sam3_set(**kw):
    with sam3_lock:
        sam3_status.update(kw)
        sam3_status["updated_at"] = datetime.now().isoformat()


def _vic_end_step(step, status, notes=""):
    """Emit a VICARIUS process_end event linked to the step's process_start.

    No-op when vicarius is disabled or the step was never started via the
    orchestrator (e.g. stale state).
    """
    if current_project is None:
        return
    s = str(step)
    st = current_project["steps"].get(s, {})
    event_id = st.pop("_vic_event_id", None)
    started = st.get("started_at")
    duration = 0.0
    if started:
        try:
            t0 = datetime.fromisoformat(started).timestamp()
            duration = max(0.0, datetime.now().timestamp() - t0)
        except Exception:
            duration = 0.0
    outputs = list(st.get("outputs", {}).values())
    step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
    vic.process_end(step_name, event_id, status, duration,
                    outputs=outputs, notes=notes)


def _sam3_drive(port, input_dir, export_dir, categories, batch_size):
    """Background driver: wait for sub-app health, configure, process, poll.

    Runs after _run_step5 launches the sub-app. Pushes progress into
    `sam3_status` which the frontend polls via /api/step/5/sam3_status.
    """
    try:
        _sam3_set(phase="waiting_for_sub_app", running=True, error=None,
                  message="Waiting for SAM3 sub-app (loading models — can take minutes)...")
        t0 = time.time()
        healthy = False
        while time.time() - t0 < 600:  # 10 min ceiling for model loading
            try:
                with urlopen(f"http://localhost:{port}/api/status", timeout=3) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if not healthy:
            _sam3_set(phase="error", running=False,
                      error="SAM3 sub-app did not become healthy within 10 minutes")
            return

        _sam3_set(phase="configuring", message="Loading sam_click_prompts and building queue...")
        body = json.dumps({
            "input_dir": input_dir,
            "export_dir": export_dir,
            "categories": categories,
            "review_batch_size": batch_size,
        }).encode()
        req = Request(f"http://localhost:{port}/api/configure", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=120) as r:
            cfg_data = json.loads(r.read())
        if cfg_data.get("error"):
            _sam3_set(phase="error", running=False,
                      error=f"configure failed: {cfg_data['error']}")
            return

        total = int(cfg_data.get("to_process") or 0)
        already = int(cfg_data.get("already_processed") or 0)
        _sam3_set(total=total,
                  message=f"{total} to segment, {already} already done. Starting...")

        # If everything is already done, skip the process call.
        if cfg_data.get("phase") == "review" or total == 0:
            _sam3_set(phase="review_ready", running=False,
                      message=f"Already segmented. Ready for review ({cfg_data.get('review_count', 0)} frames).")
            return

        # Kick off background processing on the sub-app.
        _sam3_set(phase="segmenting", message="SAM3 running...")
        req = Request(f"http://localhost:{port}/api/process", data=b"{}",
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=15):
            pass

        # Poll progress until done.
        while True:
            try:
                with urlopen(f"http://localhost:{port}/api/process_status", timeout=10) as r:
                    d = json.loads(r.read())
                _sam3_set(
                    processed=int(d.get("processed") or 0),
                    total=int(d.get("total") or 0),
                    running=bool(d.get("running", False)),
                )
                if d.get("done") or (not d.get("running") and d.get("processed", 0) >= d.get("total", 0) > 0):
                    _sam3_set(phase="review_ready", running=False,
                              message="Segmentation complete. Ready for review.")
                    return
            except Exception:
                # Transient errors are fine — the sub-app may be busy with GPU work.
                pass
            time.sleep(3)
    except Exception as e:
        _sam3_set(phase="error", running=False, error=str(e))


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        from orchestrator_config import PATHS
        return render_template(
            "index.html",
            repo_root=REPO_DIR,
            projects_dir=PATHS.get("projects_dir", os.path.join(REPO_DIR, "projects")),
            supporting_data_dir=PATHS.get("supporting_data_dir", ""),
            all_points_csv=PATHS.get("all_points_csv", ""),
            master_codes_csv=PATHS.get("master_codes_csv", ""),
        )

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------
    @app.route("/api/project/create", methods=["POST"])
    def project_create():
        global current_project
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        base_dir = data.get("base_dir", "").strip()
        if not name or not base_dir:
            return jsonify({"error": "Name and base directory are required"}), 400
        try:
            current_project = pm.create_project(name, base_dir)
            vic.note(f"[orch] project '{current_project['name']}' created at "
                     f"{current_project['project_dir']}")
            return jsonify({"success": True, "state": current_project})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/open", methods=["POST"])
    def project_open():
        global current_project
        data = request.get_json(force=True)
        project_dir = data.get("project_dir", "").strip()
        if not project_dir:
            return jsonify({"error": "Project directory is required"}), 400
        try:
            current_project = pm.load_project(project_dir)
            vic.note(f"[orch] project '{current_project['name']}' reopened")
            return jsonify({"success": True, "state": current_project})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/state")
    def project_state():
        if current_project is None:
            return jsonify({"loaded": False})
        return jsonify({"loaded": True, "state": current_project})

    @app.route("/api/project/config/<int:step>", methods=["PUT"])
    def project_config(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True)
        s = str(step)
        if s not in current_project["steps"]:
            return jsonify({"error": f"Invalid step {step}"}), 400
        current_project["steps"][s]["config"].update(data)
        pm.save_project(current_project)
        return jsonify({"success": True})

    @app.route("/api/project/quit", methods=["POST"])
    def project_quit():
        global current_project
        runner.kill_all()
        if current_project:
            vic.note(f"[orch] project '{current_project['name']}' closed")
            pm.save_project(current_project)
        current_project = None
        return jsonify({"success": True})

    # ------------------------------------------------------------------
    # Step execution endpoints
    # ------------------------------------------------------------------
    @app.route("/api/step/<int:step>/run", methods=["POST"])
    def step_run(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        s = str(step)
        st = current_project["steps"].get(s)
        if not st:
            return jsonify({"error": f"Invalid step {step}"}), 400
        if st["status"] == "locked":
            return jsonify({"error": f"Step {step} is locked"}), 400

        cfg = st["config"]
        step_dir = st["dir"]
        os.makedirs(step_dir, exist_ok=True)

        st["status"] = "running"
        st["started_at"] = datetime.now().isoformat()

        # VICARIUS process_start — stash event_id so step completion can link to it.
        step_name = f"step{step}_{pm.STEP_DIRS.get(s, '').split('_', 1)[-1] or 'stage'}"
        purpose = cfg.get("purpose") or f"{st.get('name', 'step')}" + (
            f" (project '{current_project.get('name', '')}')"
        )
        event_id = vic.process_start(step_name, purpose,
                                     inputs=[str(v) for v in cfg.values() if isinstance(v, str)],
                                     notes="")
        if event_id is not None:
            st["_vic_event_id"] = event_id
        pm.save_project(current_project)

        try:
            if step == 1:
                result = _run_step1(step, cfg, step_dir)
            elif step == 2:
                result = _run_step2(step, cfg, step_dir)
            elif step == 3:
                result = _run_step3(step, cfg, step_dir)
            elif step == 4:
                result = _run_step4(step, cfg, step_dir)
            elif step == 5:
                result = _run_step5(step, cfg, step_dir)
            elif step == 6:
                result = _run_step6(step, cfg, step_dir)
            elif step == 7:
                result = _run_step7(step, cfg, step_dir)
            elif step == 8:
                result = _run_step8(step, cfg, step_dir)
            else:
                return jsonify({"error": "Invalid step"}), 400

            if "error" in result:
                st["status"] = "error"
                _vic_end_step(step, "failed", notes=str(result.get("error", ""))[:500])
                pm.save_project(current_project)
                return jsonify(result), 500

            return jsonify({"success": True, **result})
        except Exception as e:
            st["status"] = "error"
            _vic_end_step(step, "failed", notes=str(e)[:500])
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def _run_step1(step, cfg, step_dir):
        py = PYTHON_PATHS.get(1, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[1], cfg.get("input_dir", ""), step_dir]
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[1])

    def _run_step2(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5055))
        py = PYTHON_PATHS.get(2, "python3")
        if not os.path.isfile(py):
            py = "python3"
        ap = cfg.get("all_points", "")
        mc = cfg.get("master_codes", "")
        cmd = [py, ENTRY_POINTS[2], "--port", str(port), "--all-points", ap, "--master-codes", mc]
        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[2])
        current_project["steps"]["2"]["config"]["port"] = port
        pm.save_project(current_project)
        return result

    def _run_step3(step, cfg, step_dir):
        py = PYTHON_PATHS.get(3, "python3")
        if not os.path.isfile(py):
            py = "python3"
        species = [s.strip() for s in cfg.get("target_species", "").split(",") if s.strip()]
        cmd = [
            py, ENTRY_POINTS[3],
            "--species", *species,
            "--target", str(cfg.get("target_instances", 1000)),
            "--all-points", cfg.get("all_points", ""),
            "--master-codes", cfg.get("master_codes", ""),
        ]
        if cfg.get("skip_image_check"):
            cmd.append("--skip-image-check")

        env_extra = {"TCRMP_OUTPUT_DIR": step_dir}
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[3], env_extra=env_extra)

    def _get_target_species_str():
        """Get comma-separated target species from step 3 config."""
        s3 = current_project["steps"].get("3", {}).get("config", {})
        return s3.get("target_species", "OFRA, PA, OA, OFAV, AL, MC, AA")

    def _get_remap_log_path():
        """Find the remap_log.json from step 2 output."""
        s2_dir = current_project["steps"].get("2", {}).get("dir", "")
        remap = os.path.join(s2_dir, "remap_log.json")
        if os.path.isfile(remap):
            return remap
        # Fall back to the source remap log configured in step 2
        src = current_project["steps"].get("2", {}).get("config", {}).get("remap_log_source", "")
        if src and os.path.isfile(src):
            return src
        return ""

    def _resolve_path(step_num, config_key, output_key=None):
        """Resolve a path from step config, falling back to step outputs."""
        s = str(step_num)
        step = current_project["steps"].get(s, {})
        val = step.get("config", {}).get(config_key, "")
        if not val and output_key:
            val = step.get("outputs", {}).get(output_key, "")
        return val or ""

    def _run_step4(step, cfg, step_dir):
        port = find_free_port(cfg.get("port", 5065))
        py = PYTHON_PATHS.get(4, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[4], "--port", str(port)]

        # Resolve selected_frames from step 3 (config or outputs)
        selected_frames = (
            cfg.get("selected_frames", "")
            or _resolve_path(3, "selected_frames", "selected_frames")
        )
        # Resolve all_points from step 2 (config or outputs)
        all_points = (
            cfg.get("all_points", "")
            or _resolve_path(2, "all_points", "all_points_recoded")
        )

        # Orchestrator URL the sub-app can call back to (e.g. to head-start SAM3).
        # We're inside a Flask request context so `request.host` is safe.
        host_port = request.host.split(":")[-1] if ":" in request.host else "80"
        orchestrator_url = f"http://localhost:{host_port}"

        # Pass ALL project paths so the sub-app reflects current settings
        env_extra = {
            "TCRMP_SELECTED_FRAMES": selected_frames,
            "TCRMP_EXPORT_DIR": step_dir,
            "TCRMP_CLIP_DIR": cfg.get("clip_dir", ""),
            "TCRMP_CPC_DIR": os.path.join(REPO_DIR, "output", "cpc_all"),
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_REMAP_LOG": _get_remap_log_path(),
            "TCRMP_ALL_POINTS": all_points,
            "TCRMP_ORCHESTRATOR_URL": orchestrator_url,
            "TCRMP_AUTO_START_SAM3": "1" if cfg.get("auto_start_sam3") else "",
            # These used to be user-selectable on the sub-app startup form.
            # Orchestrator-driven auto-boot bypasses that form, so we relay the
            # values here for the sub-app to pre-populate before calling configure.
            "TCRMP_REFERENCE_MODE": "1" if cfg.get("reference_mode") else "",
            "TCRMP_SHUFFLE": "1" if cfg.get("shuffle") else "",
            "TCRMP_REVIEW_BATCH_SIZE": str(cfg.get("review_batch_size") or "10"),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[4], env_extra=env_extra)
        current_project["steps"]["4"]["config"]["port"] = port
        pm.save_project(current_project)
        return result

    def _run_step5(step, cfg, step_dir):
        global sam3_thread
        port = find_free_port(cfg.get("port", 5070))
        py = PYTHON_PATHS.get(5, "python3")
        if not os.path.isfile(py):
            py = "python3"
        cmd = [py, ENTRY_POINTS[5], "--port", str(port)]

        # Resolve input_dir: step 4's output dir
        input_dir = cfg.get("input_dir", "")
        if not input_dir:
            s4_dir = current_project["steps"].get("4", {}).get("dir", "")
            if s4_dir:
                input_dir = s4_dir

        # Pass ALL project paths + SAM3 settings so sub-app reflects current config
        env_extra = {
            "TCRMP_INPUT_DIR": input_dir,
            "TCRMP_EXPORT_DIR": step_dir,
            "TCRMP_TARGET_SPECIES": _get_target_species_str(),
            "TCRMP_SAM3_DEVICE_TRACKER": cfg.get("sam3_device_tracker", ""),
            "TCRMP_SAM3_DEVICE_EXEMPLAR": cfg.get("sam3_device_exemplar", ""),
            "TCRMP_CONFIDENCE_THRESHOLD": str(cfg.get("confidence_threshold", "")),
            "TCRMP_MIN_MASK_AREA_PX": str(cfg.get("min_mask_area_px", "")),
            "TCRMP_MERGE_DISTANCE_PX": str(cfg.get("merge_distance_px", "")),
            "TCRMP_OVERLAP_STRATEGY": cfg.get("overlap_strategy", ""),
        }

        # Remove empty values so sub-app uses its own defaults
        env_extra = {k: v for k, v in env_extra.items() if v}

        result = runner.run_flask_stage(step, cmd, port, cwd=WORKING_DIRS[5], env_extra=env_extra)
        current_project["steps"]["5"]["config"]["port"] = port
        pm.save_project(current_project)

        # Kick off the orchestrator-side driver so SAM3 configures and starts
        # segmentation without the user touching the sub-app UI.
        categories = cfg.get("categories") or ["Target species only"]
        # Review batch size is user-selectable in the Step 5 panel; "all" is a
        # sentinel meaning "no batching" (the sub-app slices files[offset:+N],
        # so a huge N effectively returns everything).
        raw_batch = cfg.get("review_batch_size", "10")
        if isinstance(raw_batch, str) and raw_batch.strip().lower() == "all":
            batch_size = 999999
        else:
            try:
                batch_size = int(raw_batch)
            except (TypeError, ValueError):
                batch_size = 10
        _sam3_set(phase="launching", processed=0, total=0, running=True, error=None,
                  message="Launching SAM3 sub-app...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return result

    def _run_step6(step, cfg, step_dir):
        """Run dataset split + YOLO segmentation training via oceankind_CV.

        step_dir will receive:
            dataset/   train/valid/test split + data.yaml + test.yaml
            runs/      ultralytics training runs, one subdir per --name
        """
        # Input: step 5's export dir (contains all_images/, all_labels/, data.yaml).
        step5_dir = current_project["steps"].get("5", {}).get("dir", "")
        if not step5_dir or not os.path.isdir(step5_dir):
            return {"error": f"Step 5 output not found at {step5_dir}. Run step 5 first."}
        # Quick sanity: step 5 emits all_images + all_labels when training is meaningful.
        if not os.path.isdir(os.path.join(step5_dir, "all_images")):
            return {"error": (
                f"Step 5 dir {step5_dir} is missing all_images/. "
                "Export a batch from the SAM3 review UI to generate training data."
            )}

        # Resolve a training run name.
        run_name = (cfg.get("run_name") or "").strip()
        if not run_name:
            run_name = f"{current_project['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_project["steps"]["6"]["config"]["run_name"] = run_name

        model = cfg.get("model") or "yolo11m-seg.pt"
        epochs = str(cfg.get("epochs", 500))
        imgsz = str(cfg.get("imgsz", 512))

        # Pre-flight: ultralytics refuses batch<1 (auto) under DDP. Catch it
        # here so the user sees a helpful error instead of the subprocess
        # traceback.
        device_preview = (cfg.get("device") or "").strip()
        batch_preview = cfg.get("batch", -1)
        try:
            batch_int = int(batch_preview)
        except (TypeError, ValueError):
            batch_int = -1
        if "," in device_preview and batch_int < 1:
            gpu_count = len([d for d in device_preview.split(",") if d.strip()])
            suggested = max(2, 8 * gpu_count)
            return {"error": (
                f"Multi-GPU training (device='{device_preview}') needs an explicit "
                f"batch size — ultralytics does not support auto-batch (batch=-1) "
                f"under DDP. Set Batch to a multiple of {gpu_count} in the Step 6 "
                f"panel (e.g. {suggested}) and re-run."
            )}

        # Invoke our bash driver in "all" mode: split then train.
        cmd = [
            ENTRY_POINTS[6], "all",
            step5_dir, step_dir, run_name,
            epochs, imgsz, model,
        ]

        # Forward everything the user might have tweaked via env vars. The
        # shell driver only appends flags when the corresponding var is set,
        # so missing keys here just mean "use wrapper defaults".
        def _e(key, cfg_key, default=None):
            v = cfg.get(cfg_key, default)
            return ("", str(v)) if v is None else (key, str(v))

        env_extra_raw = dict(_e(k, ck) for k, ck in [
            ("TCRMP_STEP6_VALID", "valid_ratio"),
            ("TCRMP_STEP6_TEST", "test_ratio"),
            ("TCRMP_STEP6_MIN_SAMPLES", "min_samples"),
            # Training plumbing
            ("TCRMP_STEP6_BATCH", "batch"),
            ("TCRMP_STEP6_PATIENCE", "patience"),
            ("TCRMP_STEP6_OPTIMIZER", "optimizer"),
            ("TCRMP_STEP6_SEED", "seed"),
            ("TCRMP_STEP6_COS_LR", "cos_lr"),
            ("TCRMP_STEP6_CLOSE_MOSAIC", "close_mosaic"),
            # LR
            ("TCRMP_STEP6_LR0", "lr0"),
            ("TCRMP_STEP6_LRF", "lrf"),
            ("TCRMP_STEP6_MOMENTUM", "momentum"),
            ("TCRMP_STEP6_WEIGHT_DECAY", "weight_decay"),
            ("TCRMP_STEP6_WARMUP_EPOCHS", "warmup_epochs"),
            # Loss
            ("TCRMP_STEP6_BOX", "box"),
            ("TCRMP_STEP6_CLS", "cls"),
            ("TCRMP_STEP6_DFL", "dfl"),
            ("TCRMP_STEP6_LABEL_SMOOTHING", "label_smoothing"),
            # Color
            ("TCRMP_STEP6_HSV_H", "hsv_h"),
            ("TCRMP_STEP6_HSV_S", "hsv_s"),
            ("TCRMP_STEP6_HSV_V", "hsv_v"),
            ("TCRMP_STEP6_BGR", "bgr"),
            # Geometric
            ("TCRMP_STEP6_DEGREES", "degrees"),
            ("TCRMP_STEP6_TRANSLATE", "translate"),
            ("TCRMP_STEP6_SCALE", "scale"),
            ("TCRMP_STEP6_SHEAR", "shear"),
            ("TCRMP_STEP6_PERSPECTIVE", "perspective"),
            ("TCRMP_STEP6_FLIPUD", "flipud"),
            ("TCRMP_STEP6_FLIPLR", "fliplr"),
            # Mixing
            ("TCRMP_STEP6_MOSAIC", "mosaic"),
            ("TCRMP_STEP6_MIXUP", "mixup"),
            ("TCRMP_STEP6_CUTMIX", "cutmix"),
            ("TCRMP_STEP6_COPY_PASTE", "copy_paste"),
            ("TCRMP_STEP6_COPY_PASTE_MODE", "copy_paste_mode"),
            ("TCRMP_STEP6_ERASING", "erasing"),
            ("TCRMP_STEP6_AUTO_AUGMENT", "auto_augment"),
        ])
        env_extra = {k: v for k, v in env_extra_raw.items() if k and v not in ("", "None")}

        # Class-inclusion filter: if the user picked a subset of classes in the
        # orchestrator, forward them as a CSV env var. Empty / None / "all" means
        # "train on every class present in step 5".
        include_classes = cfg.get("include_classes")
        if isinstance(include_classes, (list, tuple)) and len(include_classes) > 0:
            csv = ",".join(str(int(x)) for x in include_classes)
            env_extra["TCRMP_STEP6_INCLUDE_CLASSES"] = csv

        # Device selection — forwarded as ultralytics `device=`. A comma-separated
        # value (e.g. "0,1") triggers DDP across those GPUs. Blank means "wrapper
        # default" (cuda:0).
        device = (cfg.get("device") or "").strip()
        if device:
            env_extra["TCRMP_STEP6_DEVICE"] = device

        current_project["steps"]["6"]["config"].update({
            "run_name": run_name,
            "model": model,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "include_classes": include_classes if isinstance(include_classes, list) else None,
            "device": device or None,
        })
        pm.save_project(current_project)
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[6], env_extra=env_extra)

    def _run_step7(step, cfg, step_dir):
        """Run ultralytics val on a trained run + emit metrics JSON + PDF report."""
        run_dir = (cfg.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return {"error": (
                "Select a training run to evaluate. Expected a path like "
                f"{current_project['steps']['6']['dir']}/runs/<run_name>/"
            )}
        # Dataset is whatever the split step wrote.
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        dataset_dir = os.path.join(step6_dir, "dataset")
        if not os.path.isdir(dataset_dir):
            return {"error": f"No dataset at {dataset_dir} — has step 6 run?"}

        py = PYTHON_PATHS.get(7)
        if not py or not os.path.isfile(py):
            return {"error": "Step 7 env not found — run TCRMPtrain_oceankindCV/setup_env.sh"}

        # Place evaluation outputs in a dedicated subdir tagged with the run name
        # so multiple evaluations don't clobber each other.
        run_name = os.path.basename(run_dir.rstrip("/"))
        out_dir = os.path.join(step_dir, f"eval_{run_name}")
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------------------
        # imgsz handling: if the user left it blank/0, auto-read the value
        # the run was trained at from its args.yaml. If they supplied one
        # that differs from the training imgsz by more than 2×, surface a
        # non-blocking warning in the response (eval still runs).
        # --------------------------------------------------------------
        trained_imgsz = None
        args_path = os.path.join(run_dir, "args.yaml")
        if os.path.isfile(args_path):
            try:
                import yaml as _y
                with open(args_path) as f:
                    _args = _y.safe_load(f) or {}
                ti = _args.get("imgsz")
                if isinstance(ti, (int, float)) and ti > 0:
                    trained_imgsz = int(ti)
            except Exception:
                trained_imgsz = None

        raw_imgsz = cfg.get("imgsz")
        try:
            user_imgsz = int(raw_imgsz) if raw_imgsz not in (None, "", 0, "0") else 0
        except (TypeError, ValueError):
            user_imgsz = 0

        warnings = []
        if user_imgsz <= 0:
            # Auto-fill from the run's args.yaml; last-resort fallback 1024.
            chosen_imgsz = trained_imgsz if trained_imgsz else 1024
            warnings.append(
                f"imgsz was blank; auto-filled from run args.yaml -> {chosen_imgsz}"
                if trained_imgsz else
                "imgsz was blank and run has no args.yaml; falling back to 1024"
            )
        else:
            chosen_imgsz = user_imgsz
            if trained_imgsz and (
                chosen_imgsz > 2 * trained_imgsz or trained_imgsz > 2 * chosen_imgsz
            ):
                warnings.append(
                    f"imgsz={chosen_imgsz} differs from training imgsz={trained_imgsz} "
                    f"by more than 2×. Metrics may be distorted — consider matching the "
                    f"training value."
                )

        cmd = [
            py, ENTRY_POINTS[7],
            "--run_dir", run_dir,
            "--dataset_dir", dataset_dir,
            "--out_dir", out_dir,
            "--split", str(cfg.get("split", "test")),
            "--imgsz", str(chosen_imgsz),
            "--batch", str(cfg.get("batch", -1)),
            "--conf", str(cfg.get("conf_threshold", 0.25)),
            "--iou", str(cfg.get("iou_threshold", 0.6)),
            "--preview_count", str(cfg.get("preview_count", 8)),
        ]
        pdf_export_dir = (cfg.get("pdf_export_dir") or "").strip()
        if pdf_export_dir:
            cmd += ["--pdf_export_dir", pdf_export_dir]

        # Persist the evaluation-output dir so the panel knows where to look.
        current_project["steps"]["7"]["config"]["run_dir"] = run_dir
        current_project["steps"]["7"]["config"]["imgsz"] = chosen_imgsz
        current_project["steps"]["7"]["outputs"]["out_dir"] = out_dir
        current_project["steps"]["7"]["outputs"]["report_md"] = os.path.join(out_dir, "report.md")
        current_project["steps"]["7"]["outputs"]["report_pdf"] = os.path.join(out_dir, "report.pdf")
        current_project["steps"]["7"]["outputs"]["metrics_json"] = os.path.join(out_dir, "metrics.json")
        pm.save_project(current_project)
        result = runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[7])
        # Attach any warnings so the UI can surface them (non-blocking).
        if warnings and isinstance(result, dict):
            existing = result.get("warnings") or []
            if isinstance(existing, list):
                result["warnings"] = existing + warnings
            else:
                result["warnings"] = warnings
        return result

    def _run_step8(step, cfg, step_dir):
        """Run model inference on a user-selected source and save overlays + manifest."""
        run_dir = (cfg.get("run_dir") or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return {"error": "Select a training run whose weights to use."}

        py = PYTHON_PATHS.get(8)
        if not py or not os.path.isfile(py):
            return {"error": "Step 8 env not found — run TCRMPtrain_oceankindCV/setup_env.sh"}

        # User-supplied run name — sanitize and stamp into the output folder.
        import re as _re
        run_name = (cfg.get("run_name") or "").strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
            out_leaf = f"{safe}_{timestamp}"
        else:
            safe = ""
            out_leaf = f"infer_{timestamp}"
        out_dir = os.path.join(step_dir, out_leaf)
        os.makedirs(out_dir, exist_ok=True)

        source_type = cfg.get("source_type", "unused")
        cmd = [
            py, ENTRY_POINTS[8],
            "--run_dir", run_dir,
            "--out_dir", out_dir,
            "--source_type", source_type,
            "--conf", str(cfg.get("conf_threshold", 0.25)),
            "--iou", str(cfg.get("iou_threshold", 0.7)),
            "--imgsz", str(cfg.get("imgsz", 512)),
            "--sample_count", str(cfg.get("sample_count", 100)),
            "--sample_mode", str(cfg.get("sample_mode", "random")),
            "--mask_alpha", str(cfg.get("mask_alpha", 0.45)),
            "--run_name", run_name or out_leaf,
            "--project_name", current_project.get("name", ""),
        ]
        if cfg.get("draw_boxes"):
            cmd.append("--draw_boxes")
        if source_type == "dir":
            source_dir = (cfg.get("source_dir") or "").strip()
            if not source_dir:
                return {"error": "source_type=dir requires source_dir"}
            cmd += ["--source_dir", source_dir]
        elif source_type in ("unused", "full"):
            # Pass the step 3 selected_frames.csv so we can exclude those for "unused".
            selected_frames = (
                current_project["steps"].get("3", {}).get("outputs", {}).get("selected_frames")
                or current_project["steps"].get("3", {}).get("config", {}).get("selected_frames")
                or ""
            )
            clip_dir = current_project["steps"].get("4", {}).get("config", {}).get("clip_dir", "")
            if selected_frames:
                cmd += ["--selected_frames", selected_frames]
            if clip_dir:
                cmd += ["--clip_dir", clip_dir]
        if cfg.get("save_overlays"):
            cmd.append("--save_overlays")
        if cfg.get("save_crops"):
            cmd.append("--save_crops")
        if cfg.get("include_pts_variant"):
            cmd.append("--include_pts_variant")

        current_project["steps"]["8"]["config"]["run_dir"] = run_dir
        current_project["steps"]["8"]["outputs"]["out_dir"] = out_dir
        current_project["steps"]["8"]["outputs"]["manifest"] = os.path.join(out_dir, "manifest.json")
        pm.save_project(current_project)
        return runner.run_cli_stage(step, cmd, cwd=WORKING_DIRS[8])

    # ------------------------------------------------------------------
    # Link existing outputs (skip running a step)
    # ------------------------------------------------------------------
    @app.route("/api/step/1/link", methods=["POST"])
    def step1_link():
        """Link existing all_points + master_codes to skip Step 1."""
        global current_project
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        data = request.get_json(force=True)
        ap_path = data.get("all_points", "").strip()
        mc_path = data.get("master_codes", "").strip()

        if not ap_path or not os.path.isfile(ap_path):
            return jsonify({"error": f"all_points file not found: {ap_path}"}), 400

        # master_codes: try to auto-detect from same directory if not given
        if not mc_path:
            mc_path = os.path.join(os.path.dirname(ap_path), "master_codes.csv")
        if not os.path.isfile(mc_path):
            return jsonify({"error": f"master_codes file not found: {mc_path}"}), 400

        step1 = current_project["steps"]["1"]
        step_dir = step1["dir"]
        os.makedirs(step_dir, exist_ok=True)

        # Symlink or copy into the project step dir
        import shutil
        for src, name in [(ap_path, os.path.basename(ap_path)), (mc_path, os.path.basename(mc_path))]:
            dst = os.path.join(step_dir, name)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        step1["outputs"]["all_points"] = ap_path
        step1["outputs"]["master_codes"] = mc_path

        # Count rows for feedback
        rows = None
        try:
            with open(ap_path) as f:
                rows = sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass

        pm.complete_step(current_project, 1)

        return jsonify({
            "success": True,
            "rows": rows,
            "state": current_project,
        })

    # ------------------------------------------------------------------
    # Step status / control
    # ------------------------------------------------------------------
    @app.route("/api/step/<int:step>/status")
    def step_status(step):
        status = runner.poll_status(step)
        # For Flask stages, add health check
        port = runner.ports.get(step)
        if port:
            status["healthy"] = runner.health_check(port)
            status["port"] = port

        # Auto-detect CLI completion
        if current_project:
            s = str(step)
            st = current_project["steps"].get(s, {})
            if st.get("status") == "running" and not status["running"]:
                if status["exit_code"] == 0:
                    _vic_end_step(step, "success")
                    pm.complete_step(current_project, step)
                    status["completed"] = True
                else:
                    st["status"] = "error"
                    _vic_end_step(step, "failed",
                                  notes=f"exit_code={status.get('exit_code')}")
                    pm.save_project(current_project)

        return jsonify(status)

    @app.route("/api/step/<int:step>/log")
    def step_log(step):
        offset = request.args.get("offset", 0, type=int)
        lines = runner.get_log(step, offset)
        return jsonify({"lines": lines, "offset": offset + len(lines)})

    @app.route("/api/step/<int:step>/done", methods=["POST"])
    def step_done(step):
        """User marks an interactive (Flask) step as complete."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill the sub-app
        runner.kill(step)
        _vic_end_step(step, "success", notes="marked done via UI")
        pm.complete_step(current_project, step)
        return jsonify({"success": True, "state": current_project})

    @app.route("/api/step/<int:step>/stop", methods=["POST"])
    def step_stop(step):
        runner.kill(step)
        if current_project:
            s = str(step)
            st = current_project["steps"].get(s, {})
            if st.get("status") == "running":
                st["status"] = "pending"
                _vic_end_step(step, "cancelled", notes="stopped via UI")
                pm.save_project(current_project)
        return jsonify({"success": True})

    @app.route("/api/step/<int:step>/reset", methods=["POST"])
    def step_reset(step):
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        # Kill any running processes for this and later steps
        last = int(pm.STEP_KEYS[-1])
        for s in range(step, last + 1):
            runner.kill(s)
        pm.reset_step(current_project, step)
        return jsonify({"success": True, "state": current_project})

    # ------------------------------------------------------------------
    # Remap endpoints
    # ------------------------------------------------------------------
    @app.route("/api/remap/browse")
    def remap_browse():
        extra = []
        if current_project:
            extra.append(current_project["project_dir"])
        logs = remap_loader.browse_remap_logs(REPO_DIR, extra_dirs=extra)
        return jsonify({"logs": logs})

    @app.route("/api/remap/load", methods=["POST"])
    def remap_load():
        data = request.get_json(force=True)
        path = data.get("path", "")
        if not os.path.isfile(path):
            return jsonify({"error": "File not found"}), 404
        try:
            log = remap_loader.load_remap_log(path)
            return jsonify({"success": True, "remap_log": log})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/remap/apply", methods=["POST"])
    def remap_apply():
        """Auto-apply remaps without launching interactive UI."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        data = request.get_json(force=True)
        remap_log_path = data.get("remap_log_path", "")
        if not os.path.isfile(remap_log_path):
            return jsonify({"error": "Remap log file not found"}), 404

        step2 = current_project["steps"]["2"]
        ap = step2["config"].get("all_points", "")
        mc = step2["config"].get("master_codes", "")
        if not ap or not mc:
            return jsonify({"error": "Step 1 outputs not linked (all_points / master_codes)"}), 400

        output_dir = step2["dir"]
        try:
            step2["status"] = "running"
            pm.save_project(current_project)

            result = remap_loader.apply_remaps(ap, mc, remap_log_path, output_dir)

            step2["config"]["remap_log_source"] = remap_log_path
            pm.complete_step(current_project, 2)

            return jsonify({"success": True, **result, "state": current_project})
        except Exception as e:
            step2["status"] = "error"
            pm.save_project(current_project)
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ------------------------------------------------------------------
    # Step 7/8: listings, reports, downloads
    # ------------------------------------------------------------------
    @app.route("/api/step/6/presets")
    def step6_list_presets():
        """List Step 6 training presets shipped in pipeline_orchestrator/presets/."""
        preset_dir = PRESET_DIRS.get("train", "")
        out = []
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append({"id": fname, "error": f"parse error: {e}"})
                    continue
                meta = data.get("_meta") or {}
                out.append({
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                })
        return jsonify({"presets": out, "dir": preset_dir})

    @app.route("/api/step/6/presets/<path:preset_id>")
    def step6_get_preset(preset_id):
        """Return the parsed params dict for one preset. preset_id is the filename."""
        # Reject any path traversal attempts — only allow plain filenames in the dir.
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("train", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        meta = data.get("_meta") or {}
        params = data.get("params") or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    # ------------------------------------------------------------------
    # Step 7: preset loader (mirrors step 6 preset endpoints) + run_info
    # ------------------------------------------------------------------
    @app.route("/api/step/7/presets")
    def step7_list_presets():
        """List Step 7 evaluation presets shipped in pipeline_orchestrator/eval_presets/."""
        preset_dir = PRESET_DIRS.get("eval", "")
        out = []
        if os.path.isdir(preset_dir):
            import yaml as _y
            for fname in sorted(os.listdir(preset_dir)):
                if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(preset_dir, fname)
                try:
                    with open(fpath) as f:
                        data = _y.safe_load(f) or {}
                except Exception as e:
                    out.append({"id": fname, "error": f"parse error: {e}"})
                    continue
                meta = data.get("_meta") or {}
                out.append({
                    "id": fname,
                    "name": meta.get("name") or fname,
                    "description": meta.get("description") or "",
                })
        return jsonify({"presets": out, "dir": preset_dir})

    @app.route("/api/step/7/presets/<path:preset_id>")
    def step7_get_preset(preset_id):
        """Return the parsed params dict for one Step 7 preset. preset_id is the filename."""
        safe_name = os.path.basename(preset_id)
        if safe_name != preset_id:
            return jsonify({"error": "invalid preset id"}), 400
        preset_dir = PRESET_DIRS.get("eval", "")
        fpath = os.path.join(preset_dir, safe_name)
        if not os.path.isfile(fpath):
            return jsonify({"error": f"preset not found: {safe_name}"}), 404
        import yaml as _y
        try:
            with open(fpath) as f:
                data = _y.safe_load(f) or {}
        except Exception as e:
            return jsonify({"error": f"parse error: {e}"}), 500
        meta = data.get("_meta") or {}
        params = data.get("params") or {}
        return jsonify({
            "id": safe_name,
            "name": meta.get("name") or safe_name,
            "description": meta.get("description") or "",
            "params": params,
        })

    @app.route("/api/step/7/run_info")
    def step7_run_info():
        """Peek at a training run's args.yaml so the Step 7 Quick-start banner
        can auto-fill `imgsz` (among a few other useful fields). Returns the
        subset of fields the UI needs; missing file -> 200 with empty values."""
        run_dir = (request.args.get("path") or "").strip()
        if not run_dir:
            return jsonify({"error": "missing ?path=<run_dir>"}), 400
        if not os.path.isdir(run_dir):
            return jsonify({"error": f"not a directory: {run_dir}"}), 404
        args_path = os.path.join(run_dir, "args.yaml")
        info = {"model": None, "imgsz": None, "epochs": None}
        if os.path.isfile(args_path):
            import yaml as _y
            try:
                with open(args_path) as f:
                    data = _y.safe_load(f) or {}
            except Exception as e:
                return jsonify({"error": f"parse error: {e}", **info}), 500
            info["model"] = data.get("model")
            info["imgsz"] = data.get("imgsz")
            info["epochs"] = data.get("epochs")
        return jsonify({"run_dir": run_dir, "args_yaml_found": os.path.isfile(args_path), **info})

    @app.route("/api/step/6/list_classes")
    def step6_list_classes():
        """Enumerate classes present in step 5's export so the user can pick
        which ones to include in training. Returns class id, name (from
        data.yaml), and per-class label-instance count from all_labels/."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step5_dir = current_project["steps"].get("5", {}).get("dir", "")
        if not step5_dir or not os.path.isdir(step5_dir):
            return jsonify({"error": f"Step 5 output not found at {step5_dir}"}), 400

        names = {}
        yaml_path = os.path.join(step5_dir, "data.yaml")
        if os.path.isfile(yaml_path):
            try:
                import yaml as _y
                with open(yaml_path) as f:
                    data = _y.safe_load(f) or {}
                raw = data.get("names") or {}
                if isinstance(raw, dict):
                    names = {int(k): v for k, v in raw.items()}
                elif isinstance(raw, list):
                    names = {i: v for i, v in enumerate(raw)}
            except Exception as e:
                return jsonify({"error": f"Could not parse {yaml_path}: {e}"}), 500

        label_dir = os.path.join(step5_dir, "all_labels")
        counts = {}
        image_counts = {}
        if os.path.isdir(label_dir):
            for fname in os.listdir(label_dir):
                fpath = os.path.join(label_dir, fname)
                seen_in_file = set()
                try:
                    with open(fpath) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                cid = int(line.split()[0])
                            except ValueError:
                                continue
                            counts[cid] = counts.get(cid, 0) + 1
                            seen_in_file.add(cid)
                except OSError:
                    continue
                for cid in seen_in_file:
                    image_counts[cid] = image_counts.get(cid, 0) + 1

        all_ids = sorted(set(names.keys()) | set(counts.keys()))
        classes = []
        for cid in all_ids:
            nm = names.get(cid)
            classes.append({
                "id": cid,
                "name": (nm if nm is not None else ""),
                "instance_count": counts.get(cid, 0),
                "image_count": image_counts.get(cid, 0),
            })
        saved = current_project["steps"].get("6", {}).get("config", {}).get("include_classes")
        return jsonify({
            "classes": classes,
            "step5_dir": step5_dir,
            "include_classes": saved,
        })

    @app.route("/api/step/6/list_runs")
    def step6_list_runs():
        """List training runs that step 7/8 can pick from."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        step6_dir = current_project["steps"].get("6", {}).get("dir", "")
        runs_dir = os.path.join(step6_dir, "runs")
        if not os.path.isdir(runs_dir):
            return jsonify({"runs": []})
        out = []
        for nm in sorted(os.listdir(runs_dir)):
            rdir = os.path.join(runs_dir, nm)
            if not os.path.isdir(rdir):
                continue
            weights_dir = os.path.join(rdir, "weights")
            best = os.path.join(weights_dir, "best.pt")
            last = os.path.join(weights_dir, "last.pt")
            has_best = os.path.isfile(best)
            has_last = os.path.isfile(last)
            # Pull a small subset of training args for the dropdown tooltip.
            args_yaml = os.path.join(rdir, "args.yaml")
            model = epochs = None
            if os.path.isfile(args_yaml):
                try:
                    import yaml as _y
                    with open(args_yaml) as f:
                        a = _y.safe_load(f) or {}
                    model = a.get("model")
                    epochs = a.get("epochs")
                except Exception:
                    pass
            out.append({
                "name": nm,
                "path": rdir,
                "has_best": has_best,
                "has_last": has_last,
                "model": model,
                "epochs": epochs,
            })
        return jsonify({"runs": out, "runs_dir": runs_dir})

    @app.route("/api/step/7/report")
    def step7_report():
        """Return the latest evaluation report (markdown + metrics JSON + pdf path)."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = current_project["steps"].get("7", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir or not os.path.isdir(out_dir):
            return jsonify({"ready": False, "message": "No evaluation has been run yet."})

        md_path = os.path.join(out_dir, "report.md")
        json_path = os.path.join(out_dir, "metrics.json")
        pdf_path = os.path.join(out_dir, "report.pdf")

        md = None
        metrics = None
        if os.path.isfile(md_path):
            with open(md_path) as f:
                md = f.read()
        if os.path.isfile(json_path):
            with open(json_path) as f:
                try:
                    metrics = json.load(f)
                except Exception:
                    metrics = None

        return jsonify({
            "ready": bool(md),
            "markdown": md,
            "metrics": metrics,
            "pdf_exists": os.path.isfile(pdf_path),
            "pdf_path": pdf_path if os.path.isfile(pdf_path) else None,
            "out_dir": out_dir,
        })

    @app.route("/api/step/7/pdf")
    def step7_pdf():
        """Serve the most recent evaluation PDF for download."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        pdf = current_project["steps"].get("7", {}).get("outputs", {}).get("report_pdf", "")
        if not pdf or not os.path.isfile(pdf):
            return jsonify({"error": "No PDF available"}), 404
        from flask import send_file
        return send_file(pdf, as_attachment=True, download_name=os.path.basename(pdf))

    @app.route("/api/step/7/copy_pdf", methods=["POST"])
    def step7_copy_pdf():
        """Copy the evaluation PDF to a user-provided folder."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        dst_dir = (data.get("dst_dir") or "").strip()
        if not dst_dir:
            return jsonify({"error": "dst_dir is required"}), 400
        pdf = current_project["steps"].get("7", {}).get("outputs", {}).get("report_pdf", "")
        if not pdf or not os.path.isfile(pdf):
            return jsonify({"error": "No PDF to copy yet"}), 404
        try:
            os.makedirs(dst_dir, exist_ok=True)
            import shutil
            dst = os.path.join(dst_dir, os.path.basename(pdf))
            shutil.copy2(pdf, dst)
            return jsonify({"ok": True, "dst": dst})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/step/8/manifest")
    def step8_manifest():
        """Return the inference gallery manifest."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        manifest_path = current_project["steps"].get("8", {}).get("outputs", {}).get("manifest", "")
        if not manifest_path or not os.path.isfile(manifest_path):
            return jsonify({"ready": False})
        with open(manifest_path) as f:
            data = json.load(f)
        data["ready"] = True
        return jsonify(data)

    @app.route("/api/step/8/image")
    def step8_image():
        """Stream a preview image from the latest inference run."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        out_dir = current_project["steps"].get("8", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir:
            return jsonify({"error": "no inference run"}), 404
        sub = request.args.get("path", "")
        # Confine to out_dir to keep this safe.
        full = os.path.realpath(os.path.join(out_dir, sub))
        if not full.startswith(os.path.realpath(out_dir)) or not os.path.isfile(full):
            return jsonify({"error": "not found"}), 404
        from flask import send_file
        return send_file(full)

    @app.route("/api/step/8/verify", methods=["POST"])
    def step8_verify():
        """Record a yes/no verdict on an inference result (quick review UI)."""
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400
        data = request.get_json(force=True) or {}
        filename = data.get("filename", "")
        verdict = data.get("verdict", "")  # yes | no | skip
        notes = data.get("notes", "")
        out_dir = current_project["steps"].get("8", {}).get("outputs", {}).get("out_dir", "")
        if not out_dir:
            return jsonify({"error": "no inference run"}), 400
        if verdict not in ("yes", "no", "skip"):
            return jsonify({"error": "verdict must be yes|no|skip"}), 400
        log_path = os.path.join(out_dir, "verifications.csv")
        is_new = not os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            if is_new:
                w.writerow(["filename", "verdict", "notes", "recorded_at"])
            w.writerow([filename, verdict, notes, datetime.now().isoformat()])
        return jsonify({"ok": True, "log_path": log_path})

    # ------------------------------------------------------------------
    # SAM3 driver status (orchestrator-driven)
    # ------------------------------------------------------------------
    @app.route("/api/step/5/sam3_status")
    def step5_sam3_status():
        with sam3_lock:
            return jsonify(dict(sam3_status))

    @app.route("/api/step/5/kick", methods=["POST"])
    def step5_kick():
        """Head-start SAM3. Called by step 4 sub-app after each batch export
        (when auto_start_sam3 is on) and by any other orchestrator-side flow
        that wants to nudge segmentation forward.

        - If step 5 sub-app is not running: launch it + start the SAM3 driver.
        - If it's running and the driver is idle/done: re-run configure+process
          so newly exported frames get picked up.
        - If the driver is still working: no-op (the sub-app's queue is rebuilt
          on the next cycle; new frames will be picked up then).
        """
        global sam3_thread
        if current_project is None:
            return jsonify({"error": "No project loaded"}), 400

        s5 = current_project["steps"].get("5")
        if not s5:
            return jsonify({"error": "Step 5 not in project"}), 400

        # Case 1: sub-app not running — do a full launch (same as Start button).
        if not runner.is_running(5):
            cfg = s5["config"]
            step_dir = s5["dir"]
            os.makedirs(step_dir, exist_ok=True)
            s5["status"] = "running"
            s5["started_at"] = datetime.now().isoformat()
            pm.save_project(current_project)
            try:
                result = _run_step5(5, cfg, step_dir)
                if "error" in result:
                    s5["status"] = "error"
                    pm.save_project(current_project)
                    return jsonify(result), 500
                return jsonify({"launched": True, **result})
            except Exception as e:
                s5["status"] = "error"
                pm.save_project(current_project)
                return jsonify({"error": str(e)}), 500

        # Case 2: sub-app running, driver thread still active — nothing to do.
        if sam3_thread is not None and sam3_thread.is_alive():
            return jsonify({"status": "already_driving"})

        # Case 3: sub-app running, driver idle — re-kick with fresh configure.
        port = runner.ports.get(5)
        if not port:
            return jsonify({"error": "No port recorded for running step 5"}), 500
        cfg = s5["config"]
        step_dir = s5["dir"]
        input_dir = cfg.get("input_dir") or current_project["steps"].get("4", {}).get("dir", "")
        categories = cfg.get("categories") or ["Target species only"]
        batch_size = int(cfg.get("review_batch_size") or 10)
        _sam3_set(phase="re_kicking", running=True, error=None,
                  message="New frames exported — re-configuring SAM3 queue...")
        sam3_thread = threading.Thread(
            target=_sam3_drive,
            args=(port, input_dir, step_dir, categories, batch_size),
            daemon=True,
        )
        sam3_thread.start()
        return jsonify({"status": "re_kicked"})

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @app.route("/api/defaults")
    def api_defaults():
        from orchestrator_config import DEFAULT_STEP_CONFIGS
        return jsonify(DEFAULT_STEP_CONFIGS)

    return app
