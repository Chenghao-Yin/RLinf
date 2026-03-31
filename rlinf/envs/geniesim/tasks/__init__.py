# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Import all task definitions to trigger registration via @register_geniesim_env
from rlinf.envs.geniesim.tasks.place_block_into_box import PlaceBlockIntoBoxEnv  # noqa: F401
from rlinf.envs.geniesim.tasks.junpu_place_workpiece import JunpuPlaceWorkpieceEnv  # noqa: F401
