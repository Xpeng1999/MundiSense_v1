#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?CH-SIMSV2 source directory required}
target_dir=${2:?derived target directory required}
mkdir -p "$target_dir"
for archive in "$source_dir"/*.zip; do
  unzip -n -q "$archive" 'Raw/*' -x '*/._*' '__MACOSX/*' -d "$target_dir"
done
