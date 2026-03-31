#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Real-time viewer for the GeneSim frame shared memory segment.
# Run in a separate terminal while the benchmark (or training) is running.
#
# Usage:
#   python3 rlinf/envs/geniesim/scripts/shm_viewer.py [--shm-name geniesim_frames] \
#       [--num-envs 1] [--width 640] [--height 480] [--fps 30]
#
# Requires: opencv-python  (pip install opencv-python)
#

import argparse
import sys
import time
from multiprocessing import shared_memory

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not found.  Install with:  pip install opencv-python")
    sys.exit(1)

_SHM_HEADER_BYTES = 4   # uint32 frame counter


def main():
    parser = argparse.ArgumentParser(description="GenieSim real-time frame viewer")
    parser.add_argument("--shm-name", default="geniesim_frames")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--width",    type=int, default=640)
    parser.add_argument("--height",   type=int, default=480)
    parser.add_argument("--num-cams", type=int, default=2,
                        help="1 = main only, 2 = main + wrist")
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="Target display frame rate")
    parser.add_argument("--save-video", type=str, default="",
                        help="If set, also write frames to this .mp4 file")
    args = parser.parse_args()

    # ---- Attach to SHM ----
    total_bytes = (_SHM_HEADER_BYTES
                   + args.num_envs * args.num_cams * args.height * args.width * 3)
    print(f"Attaching to SHM '{args.shm_name}' "
          f"({args.num_envs} envs × {args.num_cams} cams × "
          f"{args.height}×{args.width})...")
    for attempt in range(60):
        try:
            shm = shared_memory.SharedMemory(name=args.shm_name, create=False,
                                             size=total_bytes)
            # Suppress resource-tracker unlink warning on exit
            from multiprocessing import resource_tracker as _rt
            _rt.unregister(f"/{args.shm_name}", "shared_memory")
            break
        except FileNotFoundError:
            if attempt == 0:
                print("Waiting for simulation SHM to appear...")
            time.sleep(1.0)
    else:
        print(f"ERROR: SHM '{args.shm_name}' not found after 60s. "
              "Is the simulation running?")
        sys.exit(1)

    frames = np.ndarray(
        (args.num_envs, args.num_cams, args.height, args.width, 3),
        dtype=np.uint8, buffer=shm.buf, offset=_SHM_HEADER_BYTES,
    )
    counter = np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=0)
    print("Connected.  Press Q in the viewer window to quit.")

    # ---- Video writer (optional) ----
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Canvas: num_envs columns × (main + wrist) rows
        canvas_w = args.width * args.num_envs
        canvas_h = args.height * args.num_cams
        writer = cv2.VideoWriter(args.save_video, fourcc, args.fps,
                                 (canvas_w, canvas_h))
        print(f"Recording to '{args.save_video}'...")

    # ---- Display loop ----
    frame_period = 1.0 / args.fps
    prev_count = -1
    t_next = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now < t_next:
            time.sleep(max(0.0, t_next - now))
        t_next += frame_period

        curr = int(counter[0])
        # Skip display if renderer hasn't written a new frame
        if curr == prev_count:
            continue
        prev_count = curr

        # Build display canvas: rows = cameras, cols = envs
        rows = []
        for cam in range(args.num_cams):
            env_imgs = [np.copy(frames[e, cam]) for e in range(args.num_envs)]
            row_img = np.concatenate(env_imgs, axis=1)   # [H, N*W, 3]
            # Add label overlay
            label = "main" if cam == 0 else "wrist"
            cv2.putText(row_img, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            rows.append(row_img)

        canvas = np.concatenate(rows, axis=0)   # [cams*H, N*W, 3]
        # GenieSim frames are RGB; OpenCV expects BGR
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        # Frame counter overlay
        cv2.putText(canvas_bgr, f"frame {curr}", (8, canvas_bgr.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("GenieSim viewer", canvas_bgr)
        if writer:
            writer.write(canvas_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---- Cleanup ----
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Video saved to '{args.save_video}'")
    shm.close()


if __name__ == "__main__":
    main()
