# Copyright 2025 The RLinf Authors.
#
# SimContainerManager — manages the GenieSim simulation container lifecycle
# for RLinf training.
#
# Architecture:
#   HOST  GenieSimBaseEnv ──(ROS2/SHM via --network=host --ipc=host)──┐
#                                                                       │
#   CONTAINER  sim_server.py → ProcessManager → Isaac Sim + MuJoCo ───┘
#
# Handshake: the container writes {geniesim_root}/.geniesim_ready when all
# simulation processes are up.  The host polls this file.
#

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_READY_FILE    = ".geniesim_ready"
_SIM_CFG_FILE  = ".sim_server_config.json"
_PROGRESS_FILE = ".geniesim_progress"   # stage updates written by sim_server.py
_STOP_FILE     = ".geniesim_stop"       # host writes → sim_server stops processes
_START_FILE    = ".geniesim_start"      # host writes → sim_server restarts processes
_IDLE_FILE     = ".geniesim_idle"       # sim_server writes → container idle, ready to restart


class ContainerStartupError(RuntimeError):
    """Raised when the container fails to start or the sim doesn't become ready."""


class SimContainerManager:
    """
    Manages the lifecycle of the GenieSim simulation Docker container.

    Typical call sequence
    ---------------------
    mgr = SimContainerManager(container_cfg)
    mgr.ensure_running(pm_kwargs)   # start + handshake
    # ... training ...
    mgr.shutdown()                  # graceful stop

    Readiness handshake
    -------------------
    The container's sim_server.py writes {geniesim_root}/.geniesim_ready
    after all MuJoCo envs print "MuJoCoRosNode ready".  The host polls for
    this file.  On container exit before readiness, logs are scanned for
    known failure patterns and a human-readable error is raised.

    Container reuse
    ---------------
    When reuse_running=True (default), a running container whose ready file
    already exists is reused without restart.  This makes repeated
    GenieSimBaseEnv instantiations fast (e.g. during training restarts).
    """

    # ---------------------------------------------------------------------- #
    # Known startup failure patterns → human-readable messages
    # ---------------------------------------------------------------------- #
    _FAILURE_PATTERNS: List[tuple] = [
        (r"CUDA.*[Ee]rror|[Ee]rror.*CUDA|GPU.*not found|no CUDA-capable device",
         "GPU/CUDA initialization failed — run `nvidia-smi` to verify GPU accessibility"),
        (r"Failed to find display|cannot connect to X|_XSERVTransmkdir",
         "Display error — ensure `headless: true` in the sim config"),
        (r"ModuleNotFoundError.*rclpy|ImportError.*rclpy",
         "rclpy not importable — entrypoint may have failed to source ROS"),
        (r"ModuleNotFoundError.*geniesim_rl_interfaces|No module named.*geniesim_rl_interfaces",
         "geniesim_rl_interfaces not built — colcon build failed; check entrypoint logs"),
        (r"No such file or directory.*\.xml|FileNotFoundError.*\.xml",
         "MJCF file not found — check container_paths.mjcf_path in the config"),
        (r"No such file or directory.*\.usd|FileNotFoundError.*\.usd",
         "USD scene/robot file not found — check container_paths for scene_usd/robot_usd"),
        (r"[Ss]hared.?[Mm]emory.*[Ee]rror|FileExistsError.*shm|/dev/shm",
         "Shared memory conflict — a stale session may hold the SHM segment; "
         "run `ipcs -m` and `ipcrm` to clean up, or change shm_name"),
        (r"[Oo]ut [Oo]f [Mm]emory|OOMKilled|OutOfMemoryError|ENOMEM",
         "Out of memory — reduce num_envs or image resolution"),
        (r"[Pp]ermission denied",
         "Permission denied — check bind-mount ownership and entrypoint ACL setup"),
        (r"entrypoint.*failed|setup\.bash.*[Ee]rror|colcon.*[Ee]rror",
         "Container entrypoint failed — inspect `docker logs` for colcon build errors"),
        (r"ROS_DOMAIN_ID.*conflict|multiple.*ROS.*nodes.*same.*domain",
         "ROS domain ID conflict — change ros_domain_id in the config"),
    ]

    def __init__(self, container_cfg):
        """
        Parameters
        ----------
        container_cfg : omegaconf.DictConfig
            Fields:
              image            (str)  Docker image name
              name             (str)  Container name
              geniesim_root    (str)  Host path to rlinf_open_source/
              keep_alive       (bool) Don't stop container on shutdown() [default False]
              reuse_running    (bool) Reuse already-running container   [default True]
              startup_timeout_sec (int) Max seconds to wait for readiness [default 300]
              ros_domain_id    (int) ROS_DOMAIN_ID passed to container   [default 0]
              extra_docker_args (list) Extra `docker run` flags          [default []]
              container_paths  (dict) Override paths inside container    [default {}]
              isaac_cache_root (str) ~/docker/isaac-sim or equivalent    [default auto]
        """
        self.image = str(container_cfg.image)
        self.name = str(container_cfg.name)
        self.geniesim_root = Path(container_cfg.geniesim_root).expanduser().resolve()
        self.keep_alive = bool(getattr(container_cfg, "keep_alive", False))
        self.reuse_running = bool(getattr(container_cfg, "reuse_running", True))
        self.startup_timeout_sec = int(getattr(container_cfg, "startup_timeout_sec", 300))
        self.ros_domain_id = int(getattr(container_cfg, "ros_domain_id", 0))
        self.extra_docker_args: List[str] = list(
            getattr(container_cfg, "extra_docker_args", []) or []
        )
        self.container_paths: Dict[str, str] = dict(
            getattr(container_cfg, "container_paths", {}) or {}
        )
        _icr = getattr(container_cfg, "isaac_cache_root", None)
        self.isaac_cache_root: Optional[Path] = (
            Path(_icr).expanduser().resolve() if _icr
            else Path.home() / "docker" / "isaac-sim"
        )

        self._ready_file    = self.geniesim_root / _READY_FILE
        self._cfg_file      = self.geniesim_root / _SIM_CFG_FILE
        self._progress_file = self.geniesim_root / _PROGRESS_FILE
        self._stop_file     = self.geniesim_root / _STOP_FILE
        self._start_file    = self.geniesim_root / _START_FILE
        self._idle_file     = self.geniesim_root / _IDLE_FILE
        self._headless: bool = True  # updated in ensure_running()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def ensure_running(self, pm_kwargs: Dict) -> None:
        """
        Ensure the simulation container is running and ready.

        If reuse_running=True and the container is already running with an
        existing ready file, returns immediately.  Otherwise starts a fresh
        container, writes the ProcessManager config JSON, and waits for
        the readiness file.

        Parameters
        ----------
        pm_kwargs : dict
            ProcessManager.__init__() keyword arguments.  Container-side path
            overrides from container_paths are applied before writing to JSON.

        Raises
        ------
        ContainerStartupError
            If docker fails, the container exits prematurely, or the timeout
            expires.  The exception message includes a classified failure cause.
        """
        self._check_docker_available()

        status = self._get_status()

        if status == "running" and self.reuse_running:
            if self._ready_file.exists():
                # Also verify SHM is accessible — the ready file could be stale
                # (written by an old sim_server.py before Isaac Sim created SHM).
                shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
                shm_ok = self._check_shm_accessible(shm_name)
                if shm_ok:
                    print(f"[SimContainer] Reusing ready container '{self.name}'")
                    return
                print(
                    f"[SimContainer] Ready file exists but SHM '{shm_name}' is not "
                    f"accessible — restarting container (stale ready file)..."
                )
            else:
                # Container running but no ready file.
                if self._idle_file.exists():
                    # sim_server.py stopped its processes and is waiting for a
                    # start signal (keep_alive shutdown path).  Reuse the
                    # container: write new config + signal a restart.
                    print(
                        f"[SimContainer] Container '{self.name}' is idle — "
                        f"restarting sim processes...",
                        flush=True,
                    )
                    shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
                    self._cleanup_stale_shm(shm_name)
                    self._cleanup_stale_ctrl_shms(shm_name)
                    self._idle_file.unlink(missing_ok=True)
                    self._ready_file.unlink(missing_ok=True)
                    self._progress_file.unlink(missing_ok=True)
                    self._headless = bool(pm_kwargs.get("headless", True))
                    self._write_sim_config(pm_kwargs)
                    self._start_file.write_text("start")
                    self._wait_ready()
                    return
                # Container running but not idle and no ready file — started
                # manually, crashed, or still starting up without our config.
                print(
                    f"[SimContainer] Container '{self.name}' is running but has no "
                    f"ready file — stopping for a fresh start with sim_server.py..."
                )
            self._run_docker(["stop", "--time", "10", self.name], check=False)
            self._run_docker(["rm", "-f", self.name], check=False)
            status = None  # fall through to fresh start below

        # Need a fresh start — clean up any existing container first.
        if status in ("running", "paused", "created"):
            print(f"[SimContainer] Stopping existing container '{self.name}'...")
            self._run_docker(["stop", "--time", "10", self.name], check=False)

        if status is not None:
            self._run_docker(["rm", "-f", self.name], check=False)

        self._ready_file.unlink(missing_ok=True)
        shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
        self._cleanup_stale_shm(shm_name)
        self._cleanup_stale_ctrl_shms(shm_name)
        self._headless = bool(pm_kwargs.get("headless", True))
        self._progress_file.unlink(missing_ok=True)
        self._write_sim_config(pm_kwargs)
        self._start(pm_kwargs)
        self._wait_ready()

    def health_check(self) -> bool:
        """Return True if the container is running and the sim is still live."""
        if self._get_status() != "running":
            return False
        return self._ready_file.exists()

    def shutdown(self) -> None:
        """
        Gracefully stop the simulation container.

        If keep_alive=True: the container and all sim processes are left running
        so the next training run can reuse them (no cold start).  The ready file
        is intentionally NOT removed so ensure_running() recognises the state.

        If keep_alive=False: container is stopped with SIGTERM→SIGKILL (15 s
        grace period) and the ready file is cleaned up.
        """
        if self.keep_alive:
            if self._get_status() != "running":
                return
            print(
                f"[SimContainer] Stopping sim processes "
                f"(container '{self.name}' stays alive for quick restart)...",
                flush=True,
            )
            self._stop_file.write_text("stop")
            # Wait for sim_server.py to stop processes and enter idle state.
            deadline = time.time() + 60
            while time.time() < deadline:
                if self._idle_file.exists():
                    print(
                        f"[SimContainer] Sim processes stopped. "
                        f"Container '{self.name}' is idle.",
                        flush=True,
                    )
                    return
                time.sleep(0.5)
            print(
                "[SimContainer] WARNING: sim processes may not have stopped cleanly "
                "(60 s timeout waiting for idle state).",
                flush=True,
            )
            return
        print(f"[SimContainer] Shutting down simulation processes...", flush=True)
        self._ready_file.unlink(missing_ok=True)
        if self._get_status() == "running":
            print(f"[SimContainer] Stopping container '{self.name}'...", flush=True)
            self._run_docker(["stop", "--time", "15", self.name], check=False)
            print(f"[SimContainer] Container stopped.", flush=True)

    # ---------------------------------------------------------------------- #
    # Container startup
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _check_shm_accessible(shm_name: str) -> bool:
        """Return True if /dev/shm/<shm_name> exists AND is readable by this process."""
        shm_path = Path("/dev/shm") / shm_name
        if not shm_path.exists():
            return False
        try:
            shm_path.open("rb").close()
            return True
        except PermissionError:
            return False

    def _cleanup_stale_shm(self, shm_name: str) -> None:
        """
        Remove a stale POSIX shared memory segment from /dev/shm if present.

        Isaac Sim creates the SHM as uid 1234 (0o600).  If the previous
        container exited uncleanly, the segment lingers and causes either a
        FileExistsError inside the new container or a PermissionError on the
        host.  sim_server.py also does this cleanup from inside the container
        (as uid 1234), but we attempt it here first as a best-effort.

        When direct unlink fails due to permissions, a temporary container is
        used so that uid 1234 can remove the file.
        """
        shm_path = Path("/dev/shm") / shm_name
        if not shm_path.exists():
            return
        try:
            shm_path.unlink()
            print(f"[SimContainer] Removed stale SHM '{shm_name}'")
        except PermissionError:
            owner_uid = shm_path.stat().st_uid
            print(
                f"[SimContainer] Stale SHM '{shm_name}' owned by uid {owner_uid} — "
                f"removing via temporary container..."
            )
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--ipc", "host",
                    "--user", str(owner_uid),
                    "--entrypoint", "/bin/rm",
                    self.image,
                    f"/dev/shm/{shm_name}",
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"[SimContainer] Stale SHM '{shm_name}' removed via container")
            else:
                print(
                    f"[SimContainer] WARNING: could not remove stale SHM '{shm_name}': "
                    f"{result.stderr.strip()} — "
                    f"run `sudo rm /dev/shm/{shm_name}` if startup fails"
                )

    def _cleanup_stale_ctrl_shms(self, shm_name: str) -> None:
        """Remove stale per-env ctrl POSIX SHM segments ({shm_name}_ctrl_N)."""
        shm_dir = Path("/dev/shm")
        prefix = f"{shm_name}_ctrl_"
        for entry in shm_dir.iterdir():
            if entry.name.startswith(prefix):
                try:
                    entry.unlink()
                    print(f"[SimContainer] Removed stale ctrl SHM '{entry.name}'")
                except PermissionError:
                    owner_uid = entry.stat().st_uid
                    result = subprocess.run(
                        [
                            "docker", "run", "--rm",
                            "--ipc", "host",
                            "--user", str(owner_uid),
                            "--entrypoint", "/bin/rm",
                            self.image,
                            f"/dev/shm/{entry.name}",
                        ],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        print(f"[SimContainer] Removed stale ctrl SHM '{entry.name}' via container")
                    else:
                        print(
                            f"[SimContainer] WARNING: could not remove ctrl SHM "
                            f"'{entry.name}': {result.stderr.strip()}"
                        )

    @staticmethod
    def _ensure_vulkan_icd() -> None:
        """
        Verify that the Vulkan ICD paths in the CDI spec exist on the host.

        The CDI spec (/var/run/cdi/nvidia.yaml) is generated by nvidia-ctk and
        may embed stale paths (e.g. /etc/vulkan/icd.d/nvidia_icd.json).  If the
        file is absent but a valid ICD exists elsewhere (e.g. /usr/share/vulkan/
        icd.d/), runc will fail at container init with "no such file or directory".

        Fix: regenerate the CDI spec so it picks up the correct host paths.
        Requires sudo (the spec lives in /var/run/cdi/ owned by root).
        """
        _CDI_SPEC = Path("/var/run/cdi/nvidia.yaml")
        _ICD_CANDIDATES = [
            Path("/etc/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/local/share/vulkan/icd.d/nvidia_icd.json"),
        ]

        # Read CDI spec and collect Vulkan ICD host paths referenced there.
        if not _CDI_SPEC.exists():
            return
        try:
            spec_text = _CDI_SPEC.read_text()
        except OSError:
            return

        missing = [
            p for p in re.findall(r"hostPath:\s*(\S+nvidia_icd[^\s]*)", spec_text)
            if not Path(p).exists()
        ]
        if not missing:
            return  # all CDI-referenced ICD paths exist — nothing to do

        # At least one referenced ICD path is missing.  Check that a valid ICD
        # exists somewhere on the host before attempting a fix.
        valid_icd = next((p for p in _ICD_CANDIDATES if p.exists()), None)
        if valid_icd is None:
            print(
                "[SimContainer] WARNING: Vulkan ICD paths in CDI spec are missing "
                f"({missing}) and no valid ICD found in standard locations. "
                "GPU rendering may fail."
            )
            return

        print(
            f"[SimContainer] CDI spec references missing Vulkan ICD path(s): {missing}\n"
            f"[SimContainer] Valid ICD found at: {valid_icd}\n"
            "[SimContainer] Regenerating CDI spec via: "
            "sudo nvidia-ctk cdi generate --output /var/run/cdi/nvidia.yaml"
        )
        result = subprocess.run(
            ["sudo", "nvidia-ctk", "cdi", "generate",
             "--output", str(_CDI_SPEC)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("[SimContainer] CDI spec regenerated successfully.")
            return

        print(
            f"[SimContainer] CDI spec regeneration failed "
            f"(rc={result.returncode}): {result.stderr.strip()}\n"
            "[SimContainer] Falling back to symlink creation..."
        )
        icd_dir = Path("/etc/vulkan/icd.d")
        icd_link = icd_dir / "nvidia_icd.json"
        result2 = subprocess.run(
            ["sudo", "mkdir", "-p", str(icd_dir)],
            capture_output=True, text=True,
        )
        if result2.returncode == 0:
            result3 = subprocess.run(
                ["sudo", "ln", "-s", str(valid_icd), str(icd_link)],
                capture_output=True, text=True,
            )
            if result3.returncode == 0:
                print(f"[SimContainer] Symlink created: {icd_link} -> {valid_icd}")
                return
        print(
            "[SimContainer] WARNING: automatic fix failed. "
            "To fix manually, run ONE of:\n"
            "  sudo nvidia-ctk cdi generate --output /var/run/cdi/nvidia.yaml\n"
            "Or:\n"
            f"  sudo mkdir -p /etc/vulkan/icd.d\n"
            f"  sudo ln -s {valid_icd} /etc/vulkan/icd.d/nvidia_icd.json"
        )

    def _start(self, pm_kwargs: Optional[Dict] = None) -> None:
        self._ensure_vulkan_icd()
        self._start_isaac(pm_kwargs or {})

    def _start_isaac(self, pm_kwargs: Dict) -> None:
        # Resolve symlinks for bind mounts (docker requires real paths).
        main_path = (self.geniesim_root / "main").resolve()
        g2sim_path = (self.geniesim_root / "g2sim").resolve()

        mounts: List[str] = [
            f"{self.geniesim_root}:/geniesim/main:rw",
            f"{main_path}:/geniesim/main/main:rw",
            f"{g2sim_path}:/geniesim/main/g2sim:rw",
        ]

        # Isaac Sim cache dirs (optional; skip if not present on host).
        _isaac_cache_map = {
            "pkg":                "/isaac-sim/.local/share/ov/pkg",
            "cache/main":         "/isaac-sim/.cache",
            "cache/computecache": "/isaac-sim/.nv/ComputeCache",
            "logs":               "/isaac-sim/.nvidia-omniverse/logs",
            "config":             "/isaac-sim/.nvidia-omniverse/config",
            "data":               "/isaac-sim/.local/share/ov/data",
        }
        for rel, container_dst in _isaac_cache_map.items():
            host_path = self.isaac_cache_root / rel
            if host_path.exists():
                mounts.append(f"{host_path}:{container_dst}:rw")

        env_vars: List[str] = [
            "SIM_REPO_ROOT=/geniesim/main/main",
            f"ROS_DOMAIN_ID={self.ros_domain_id}",
            "ROS_LOCALHOST_ONLY=1",
            "GENIESIM_ROS_WS_INSTALL=/geniesim/ros_ws_build/install",
        ]

        sim_server_cmd = (
            f"python3 /geniesim/main/main/source/geniesim/rl/scripts/sim_server.py"
            f" --config-json /geniesim/main/{_SIM_CFG_FILE}"
            f" --ready-file /geniesim/main/{_READY_FILE}"
        )

        cmd: List[str] = [
            "run", "--detach",
            "--name", self.name,
            "--network", "host",
            "--ipc",     "host",   # share /dev/shm so SHM frames are visible on host
            "--gpus",    "all",
            "--device",  "/dev/input:/dev/input",
        ]

        # Auto X11 forwarding when headless=False (set via `headless: false` in
        # the YAML config).  No manual extra_docker_args needed by the user.
        _display = os.environ.get("DISPLAY", "")
        if not self._headless:
            if _display:
                cmd += ["-e", f"DISPLAY={_display}",
                        "-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
                subprocess.run(["xhost", "+local:docker"],
                               capture_output=True, check=False)
            else:
                print(
                    "[SimContainer] WARNING: headless=false but $DISPLAY is not set. "
                    "Isaac Sim 3D window will not appear.  Set the DISPLAY env var "
                    "(e.g. export DISPLAY=:0) before launching, or set headless=true.",
                    flush=True,
                )

        for m in mounts:
            cmd += ["-v", m]
        for e in env_vars:
            cmd += ["-e", e]
        cmd += self.extra_docker_args
        cmd += [
            self.image,
            "/entrypoint_geniesim_rlinf.sh",
            "bash", "-c", sim_server_cmd,
        ]

        _vis = "headless" if self._headless else f"display={_display or '(no $DISPLAY)'}"
        print(
            f"[SimContainer] Starting '{self.name}' from image '{self.image}'\n"
            f"[SimContainer]   network=host  ipc=host  gpus=all  mode={_vis}"
        )
        self._run_docker(cmd)

    def _wait_ready(self) -> None:
        """
        Block until {geniesim_root}/.geniesim_ready is written or timeout.

        Provides live progress feedback:
          - Stage labels from .geniesim_progress (written by sim_server.py)
          - Relevant lines from docker logs (Isaac Sim / colcon build output)
          - Elapsed/remaining time every 5 s so users know the sim isn't frozen

        Typical timing on a warm cache (Isaac Sim pkg already cached):
          0-5s:   Docker start + entrypoint (colcon build already cached)
          5-15s:  MuJoCo physics nodes launch
          15-130s: Isaac Sim GPU init + USD scene load  ← main wait
        """
        _STAGE_MSGS = {
            "mujoco_launching": "Launching MuJoCo physics nodes...",
            "mujoco_ready":     "MuJoCo physics nodes ready.",
            "isaac_launching":  "Launching Isaac Sim renderer (cold start ~2 min)...",
            "isaac_loading":    "Isaac Sim loading USD scene and GPU renderer...",
            "isaac_ready":      "Isaac Sim renderer ready — finalising...",
        }
        # Isaac Sim log lines that are worth surfacing to the user.
        # These tell users what Isaac Sim is doing during the long silent period.
        _ISAAC_LOG_PATTERNS = [
            "entrypoint",
            "geniesim_rl_interfaces",
            "Installing geniesim",
            "sim_server",
            "ProcessManager",
            "MuJoCo",
            "ready",
            "Isaac Sim",
            "Loading",
            "Startup",
            "startup",
            "RTX",
            "renderer",
            "SHM",
            "shm",
        ]

        deadline = time.time() + self.startup_timeout_sec
        t_start = time.time()
        last_stage = ""
        last_heartbeat = time.time()
        _HEARTBEAT_SEC = 5.0   # progress update every 5 s (was 10 s)

        # Track which container log lines have already been shown to the user.
        _shown_log_lines: set = set()

        print(f"[SimContainer] Waiting for simulation readiness "
              f"(timeout={self.startup_timeout_sec}s)...", flush=True)

        while time.time() < deadline:
            if self._ready_file.exists():
                elapsed = time.time() - t_start
                print(f"[SimContainer] Simulation ready ({elapsed:.0f}s elapsed)", flush=True)
                return

            # ---- Stage label from sim_server.py ----
            try:
                stage = self._progress_file.read_text().strip()
                if stage and stage != last_stage:
                    msg = _STAGE_MSGS.get(stage, stage)
                    elapsed = time.time() - t_start
                    print(f"[SimContainer]   › {msg}  ({elapsed:.0f}s)", flush=True)
                    last_stage = stage
                    last_heartbeat = time.time()   # reset heartbeat on real progress
            except (FileNotFoundError, OSError):
                pass

            # ---- Container exit check ----
            status = self._get_status()
            if status != "running":
                raise ContainerStartupError(
                    f"Container '{self.name}' exited unexpectedly "
                    f"(status={status!r}).\n\n"
                    + self._classify_failure()
                )

            # ---- Tail container logs for new relevant lines ----
            try:
                log_tail = self._get_logs(tail=40)
                for line in log_tail.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped in _shown_log_lines:
                        continue
                    if any(pat.lower() in stripped.lower() for pat in _ISAAC_LOG_PATTERNS):
                        _shown_log_lines.add(stripped)
                        print(f"[SimContainer]     {stripped}", flush=True)
            except Exception:
                pass

            # ---- Periodic heartbeat (elapsed / estimated remaining) ----
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_SEC:
                elapsed = now - t_start
                remaining = self.startup_timeout_sec - elapsed
                if last_stage in ("", "mujoco_launching"):
                    hint = "entrypoint + MuJoCo init..."
                elif last_stage == "mujoco_ready":
                    hint = "waiting for Isaac Sim renderer to start..."
                elif last_stage in ("isaac_launching", "isaac_loading"):
                    hint = "Isaac Sim GPU init + USD scene load (this is the long part)..."
                else:
                    hint = "finalising..."
                print(
                    f"[SimContainer]   › {hint}  "
                    f"elapsed={elapsed:.0f}s  remaining≤{remaining:.0f}s",
                    flush=True,
                )
                last_heartbeat = now

            time.sleep(1.0)

        raise ContainerStartupError(
            f"Simulation not ready after {self.startup_timeout_sec}s.\n\n"
            + self._classify_failure()
        )

    # ---------------------------------------------------------------------- #
    # Config serialisation
    # ---------------------------------------------------------------------- #

    def _write_sim_config(self, pm_kwargs: Dict) -> None:
        """
        Write ProcessManager config JSON with container-side path overrides.
        The file lands in geniesim_root/ which is bind-mounted into the container.
        """
        cfg = dict(pm_kwargs)
        cfg.update(self.container_paths)
        self._cfg_file.write_text(json.dumps(cfg, indent=2))

    @staticmethod
    def pm_kwargs_from_vec_cfg(vec_cfg) -> Dict:
        """
        Extract ProcessManager.__init__() kwargs from a GenieSimVectorEnvConfig.

        These are the fields forwarded to sim_server.py via JSON.  Fields
        specific to the training wrapper (enable_reward, max_episode_steps,
        attach_to_running, …) are excluded.
        """
        kwargs = {
            "num_envs":        vec_cfg.num_envs,
            "mjcf_path":       vec_cfg.mjcf_path,
            "scene_usd":       vec_cfg.scene_usd,
            "robot_usd":       vec_cfg.robot_usd,
            "robot_prim":      vec_cfg.robot_prim,
            "shm_name":        vec_cfg.shm_name,
            "physics_hz":      vec_cfg.physics_hz,
            "render_hz":       vec_cfg.render_hz,
            "cam_width":       vec_cfg.cam_width,
            "cam_height":      vec_cfg.cam_height,
            "main_cam_prim":   vec_cfg.main_cam_prim,
            "wrist_cam_prim":  vec_cfg.wrist_cam_prim,
            "headless":        vec_cfg.headless,
            "ros_domain_id":   vec_cfg.ros_domain_id,
            "isaac_python":    vec_cfg.isaac_python,
            "mujoco_python":   vec_cfg.mujoco_python or "",
            "task_name":       vec_cfg.task_name,
            "robot_type":      vec_cfg.robot_type,
            "task_instance_id": vec_cfg.task_instance_id,
            "state_joint_offset": vec_cfg.state_joint_offset,
            "ctrl_offset":     vec_cfg.ctrl_offset,
            "ctrl_offset_r":   getattr(vec_cfg, "ctrl_offset_r", -1),
            "state_dim":       vec_cfg.state_dim,
            "action_dim":      vec_cfg.action_dim,
            "control_mode":    vec_cfg.control_mode,
            "gripper_ctrl_l":  vec_cfg.gripper_ctrl_l,
            "gripper_ctrl_r":  vec_cfg.gripper_ctrl_r,
            "ee_body_l":       vec_cfg.ee_body_l,
            "ee_body_r":       vec_cfg.ee_body_r,
            "ik_max_iter":     vec_cfg.ik_max_iter,
            "ik_damp":         vec_cfg.ik_damp,
            "randomization_cfg_json": vec_cfg.randomization_cfg_json,
            "reset_ee_r_json": getattr(vec_cfg, "reset_ee_r_json", ""),
            "seed":            getattr(vec_cfg, "seed", 42),
        }
        return kwargs

    # ---------------------------------------------------------------------- #
    # Docker helpers
    # ---------------------------------------------------------------------- #

    def _run_docker(self, args: List[str], check: bool = True) -> str:
        result = subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise ContainerStartupError(
                f"`docker {args[0]}` failed (rc={result.returncode}):\n"
                + (result.stderr or result.stdout or "(no output)").strip()
            )
        return result.stdout.strip()

    def _get_status(self) -> Optional[str]:
        """Return container State.Status or None if the container doesn't exist."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", self.name],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _get_logs(self, tail: int = 150) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self.name],
            capture_output=True,
            text=True,
        )
        return (result.stdout + result.stderr).strip()

    @staticmethod
    def _check_docker_available() -> None:
        result = subprocess.run(["docker", "info"], capture_output=True)
        if result.returncode != 0:
            raise ContainerStartupError(
                "Docker daemon is not running or not accessible. "
                "Start Docker and ensure the current user is in the `docker` group."
            )

    # ---------------------------------------------------------------------- #
    # Failure diagnosis
    # ---------------------------------------------------------------------- #

    def _classify_failure(self) -> str:
        """
        Scan container logs for known failure patterns and return a
        human-readable explanation with the relevant log tail.
        """
        logs = self._get_logs(tail=200)
        for pattern, message in self._FAILURE_PATTERNS:
            if re.search(pattern, logs, re.IGNORECASE):
                return (
                    f"Likely cause: {message}\n\n"
                    f"--- Last container logs ---\n{logs[-3000:]}"
                )
        return (
            f"Cause unknown — no recognised failure pattern found.\n\n"
            f"--- Last container logs ---\n{logs[-3000:]}"
        )
