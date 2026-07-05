#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run with sudo:"
    echo "  sudo $0"
    exit 1
fi

key="/var/lib/shim-signed/mok/MOK.der"

if [[ ! -f "${key}" ]]; then
    echo "MOK certificate not found: ${key}"
    exit 1
fi

echo "[1/3] Secure Boot state"
mokutil --sb-state || true

echo "[2/3] Testing whether local DKMS signing key is already enrolled"
test_output="$(mokutil --test-key "${key}" 2>&1 || true)"
echo "${test_output}"
if echo "${test_output}" | grep -qi "already enrolled"; then
    echo "MOK is already enrolled."
    exit 0
fi

echo "[3/3] Importing local DKMS signing key into MOK"
echo
echo "You will be asked to create a temporary MOK enrollment password."
echo "After reboot, the blue MOK Manager screen will ask for this password."
echo
mokutil --import "${key}"

echo
echo "Next steps:"
echo "  1. sudo reboot"
echo "  2. In the blue MOK Manager screen:"
echo "     Enroll MOK -> Continue -> Yes -> enter the password -> Reboot"
echo "  3. After Linux boots, run:"
echo "     sudo modprobe nvidia_fs"
echo "     /usr/local/cuda/gds/tools/gdscheck -p"
