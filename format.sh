#!/usr/bin/env bash
set -e

black \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/forrest \
  source/isaaclab/isaaclab/tendons \
  scripts/tendons \
  source/isaaclab_assets/isaaclab_assets/robots/forrest.py