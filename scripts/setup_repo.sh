#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

ENV_NAME="sim"
URDFHEIM_DIR=""
SKIP_CONDA=0
SKIP_INSTALL=0
SKIP_PRE_COMMIT=0
SKIP_SYMLINK=0
SKIP_USD_PATCH=0
SKIP_IDE_PATHS=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Set up IsaacNext after cloning.

Options:
  --env-name NAME       Conda environment name. Default: sim
  --urdfheim-dir PATH   Path to urdfheim. Default: ../urdfheim relative to this repo
  --skip-conda          Do not create/update the conda environment
  --skip-install        Do not install Isaac Lab source extensions
  --skip-pre-commit     Do not install pre-commit hooks
  --skip-symlink        Do not create/update the Forrest USD symlink
  --skip-usd-patch      Do not patch Forrest USD virtual visual reference targets
  --skip-ide-paths      Do not configure Isaac Sim paths for PyCharm/static analysis
  -h, --help            Show this help
EOF
}

log() {
    echo "[setup] $*"
}

error() {
    echo "[setup][ERROR] $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            [[ $# -ge 2 ]] || error "--env-name requires a value"
            ENV_NAME="$2"
            shift 2
            ;;
        --urdfheim-dir)
            [[ $# -ge 2 ]] || error "--urdfheim-dir requires a value"
            URDFHEIM_DIR="$2"
            shift 2
            ;;
        --skip-conda)
            SKIP_CONDA=1
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --skip-pre-commit)
            SKIP_PRE_COMMIT=1
            shift
            ;;
        --skip-symlink)
            SKIP_SYMLINK=1
            shift
            ;;
        --skip-usd-patch)
            SKIP_USD_PATCH=1
            shift
            ;;
        --skip-ide-paths)
            SKIP_IDE_PATHS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${URDFHEIM_DIR}" ]]; then
    URDFHEIM_DIR="$(cd "${REPO_ROOT}/.." && pwd)/urdfheim"
else
    URDFHEIM_DIR="$(realpath "${URDFHEIM_DIR}")"
fi

setup_symlink() {
    local target_dir="${URDFHEIM_DIR}/complex/forrest_isaac_description"
    local target_usd="${target_dir}/urdf/forrest_isaac/forrest_isaac.usd"
    local link_dir="${REPO_ROOT}/symlinks"
    local link_path="${link_dir}/forrest_ws"

    [[ -d "${URDFHEIM_DIR}" ]] || error "urdfheim not found at: ${URDFHEIM_DIR}"
    [[ -d "${target_dir}" ]] || error "Forrest USD directory not found: ${target_dir}"
    [[ -f "${target_usd}" ]] || error "Forrest USD file not found: ${target_usd}"

    mkdir -p "${link_dir}"

    if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
        error "${link_path} exists but is not a symlink. Move it away before running setup."
    fi

    ln -sfn "${target_dir}" "${link_path}"
    log "Forrest USD symlink: ${link_path} -> ${target_dir}"
}

setup_conda_env() {
    command -v conda >/dev/null 2>&1 || error "conda was not found on PATH"

    # Make `conda activate` available in non-interactive shells.
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "Updating conda environment: ${ENV_NAME}"
        conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml" --prune
    else
        log "Creating conda environment: ${ENV_NAME}"
        conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml"
    fi
}

install_extensions() {
    log "Installing Isaac Lab source extensions into conda env: ${ENV_NAME}"
    conda run -n "${ENV_NAME}" "${REPO_ROOT}/isaaclab.sh" -i
}

install_pre_commit() {
    log "Installing pre-commit Git hooks"
    conda run -n "${ENV_NAME}" pre-commit install
}

setup_ide_paths() {
    log "Configuring Isaac Sim IDE paths for conda env: ${ENV_NAME}"
    "${REPO_ROOT}/scripts/setup_pycharm_isaacsim_paths.sh" "${ENV_NAME}"
}

patch_forrest_usd() {
    log "Patching Forrest USD virtual visual references"
    conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/tools/patch_forrest_usd.py"
}

cd "${REPO_ROOT}"

if [[ "${SKIP_SYMLINK}" -eq 0 ]]; then
    setup_symlink
else
    log "Skipping Forrest USD symlink"
fi

if [[ "${SKIP_CONDA}" -eq 0 ]]; then
    setup_conda_env
else
    log "Skipping conda environment setup"
fi

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
    install_extensions
else
    log "Skipping Isaac Lab extension install"
fi

if [[ "${SKIP_PRE_COMMIT}" -eq 0 ]]; then
    install_pre_commit
else
    log "Skipping pre-commit hook install"
fi

if [[ "${SKIP_USD_PATCH}" -eq 0 ]]; then
    patch_forrest_usd
else
    log "Skipping Forrest USD patch"
fi

if [[ "${SKIP_IDE_PATHS}" -eq 0 ]]; then
    setup_ide_paths
else
    log "Skipping Isaac Sim IDE path setup"
fi

log "Setup complete."
log "Next shell command: conda activate ${ENV_NAME}"
