#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

ENV_NAME="${1:-sim}"

log() {
    echo "[pycharm-paths] $*"
}

error() {
    echo "[pycharm-paths][ERROR] $*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || error "conda was not found on PATH"

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
SITE_PACKAGES="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])')"
ISAACSIM_ROOT="${SITE_PACKAGES}/isaacsim"
EXTS_ROOT="${ISAACSIM_ROOT}/exts"
EXTSCACHE_ROOT="${ISAACSIM_ROOT}/extscache"
PTH_FILE="${SITE_PACKAGES}/isaacsim-ide-paths.pth"
BOOTSTRAP_FILE="${SITE_PACKAGES}/isaacsim_ide_paths.py"

[[ -d "${ISAACSIM_ROOT}" ]] || error "Isaac Sim package root not found: ${ISAACSIM_ROOT}"

TMP_FILE="$(mktemp)"

# Hand-written Isaac Sim extension roots used by static analysis.
if [[ -d "${EXTS_ROOT}" ]]; then
    find "${EXTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort >> "${TMP_FILE}"
fi

# Extension-cache roots that expose importable namespaces such as pxr, omni, carb, isaacsim, and usdrt.
if [[ -d "${EXTSCACHE_ROOT}" ]]; then
    find "${EXTSCACHE_ROOT}" -mindepth 1 -maxdepth 1 -type d | while read -r ext_dir; do
        if [[ -d "${ext_dir}/pxr" || -d "${ext_dir}/omni" || -d "${ext_dir}/carb" || -d "${ext_dir}/isaacsim" || -d "${ext_dir}/usdrt" ]]; then
            echo "${ext_dir}"
        fi
    done | sort >> "${TMP_FILE}"
fi

sort -u "${TMP_FILE}" > "${PTH_FILE}"
rm -f "${TMP_FILE}"

cat >> "${PTH_FILE}" <<EOF
import isaacsim_ide_paths
EOF

cat > "${BOOTSTRAP_FILE}" <<EOF
"""Expose Isaac Sim extension packages to plain Python and PyCharm."""

from __future__ import annotations

from pathlib import Path


def _extend_regular_package(package_name: str, extension_root: Path) -> None:
    try:
        package = __import__(package_name)
    except Exception:
        return

    package_dir = extension_root / package_name
    if package_dir.is_dir():
        package_path = str(package_dir)
        if package_path not in package.__path__:
            package.__path__.append(package_path)


_roots_file = Path(__file__).with_name("isaacsim-ide-paths.pth")
for _line in _roots_file.read_text().splitlines():
    if not _line or _line.startswith("import "):
        continue
    _root = Path(_line)
    _extend_regular_package("isaacsim", _root)
EOF

log "Wrote Python path file: ${PTH_FILE}"
log "Wrote package bootstrap: ${BOOTSTRAP_FILE}"
log "Added $(wc -l < "${PTH_FILE}") Isaac Sim extension roots"

# Some pxr modules import native USD libraries. Isaac Sim adds these at runtime, but plain Python/PyCharm do not.
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/isaacsim_usd_libs.sh" <<EOF
#!/usr/bin/env bash

export _ISAACSIM_PREV_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${EXTSCACHE_ROOT}/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:${EXTSCACHE_ROOT}/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin/deps:\${LD_LIBRARY_PATH:-}"
EOF

cat > "${DEACTIVATE_DIR}/isaacsim_usd_libs_unset.sh" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${_ISAACSIM_PREV_LD_LIBRARY_PATH+x}" ]]; then
    export LD_LIBRARY_PATH="${_ISAACSIM_PREV_LD_LIBRARY_PATH}"
    unset _ISAACSIM_PREV_LD_LIBRARY_PATH
fi
EOF

log "Wrote conda activation hooks for USD shared libraries"
log "Restart PyCharm or invalidate caches so it re-indexes the interpreter paths."
