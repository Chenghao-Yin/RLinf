# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
GeneSim RL task: place_block_into_box

Sparse reward: +1.0 when the block is placed inside the box (ADER Inside action succeeds).
"""

from rlinf.envs.geniesim import register_geniesim_env
from rlinf.envs.geniesim.geniesim_env import GenieSimBaseEnv


@register_geniesim_env("place_block_into_box")
class PlaceBlockIntoBoxEnv(GenieSimBaseEnv):
    """
    Minimal subclass for place_block_into_box task.
    All logic is inherited from GenieSimBaseEnv; the ADER action config is read
    from the task JSON specified in cfg.init_params.task_file.
    """
    pass
