#!/usr/bin/env bash
set -euo pipefail

sync
echo 3 > /proc/sys/vm/drop_caches
