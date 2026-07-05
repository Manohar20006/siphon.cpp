#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run with sudo:"
    echo "  sudo $0"
    exit 1
fi

echo "[1/6] Loading nvidia_fs now"
modprobe nvidia_fs
lsmod | grep -E '^nvidia_fs\b'

echo "[2/6] Ensuring nvidia_fs loads on boot"
printf 'nvidia_fs\n' > /etc/modules-load.d/nvidia-fs.conf

echo "[3/6] Backing up /etc/default/grub"
backup="/etc/default/grub.bak.$(date +%Y%m%d-%H%M%S)"
cp /etc/default/grub "${backup}"
echo "Backup written to ${backup}"

echo "[4/6] Adding Intel IOMMU-off flags to GRUB_CMDLINE_LINUX_DEFAULT"
python3 - <<'PY'
from pathlib import Path

path = Path("/etc/default/grub")
text = path.read_text()
needed = ["intel_iommu=off", "iommu=off"]
lines = text.splitlines()
out = []
done = False
for line in lines:
    if line.startswith("GRUB_CMDLINE_LINUX_DEFAULT="):
        prefix, value = line.split("=", 1)
        quote = '"'
        current = value.strip()
        if current.startswith('"') and current.endswith('"'):
            args = current[1:-1].split()
        else:
            args = current.strip('"').split()
        for item in needed:
            if item not in args:
                args.append(item)
        out.append(f'{prefix}="{ " ".join(args) }"')
        done = True
    else:
        out.append(line)
if not done:
    out.append('GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off iommu=off"')
path.write_text("\n".join(out) + "\n")
PY

echo "[5/6] Updating GRUB"
update-grub

echo "[6/6] Current GDS check before reboot"
/usr/local/cuda/gds/tools/gdscheck -p || true

echo
echo "Next step: reboot, then run:"
echo "  lsmod | grep nvidia_fs"
echo "  /usr/local/cuda/gds/tools/gdscheck -p"
echo "  CUFILE_ENV_PATH_JSON=/home/manohar/Desktop/inference/qwen.cpp/cufile-gds-p2p.json /usr/local/cuda/gds/tools/gdscheck -p"
