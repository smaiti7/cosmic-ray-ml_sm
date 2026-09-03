#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/cosmic_array_dataset.npz}"
expected_bytes=1908350470
mkdir -p "$(dirname "$destination")"

if [[ -f "$destination" ]] && [[ "$(stat -c %s "$destination")" -eq "$expected_bytes" ]]; then
  echo "Dataset already downloaded: $destination"
  exit 0
fi

wget --continue --progress=bar:force:noscroll \
  -O "$destination" \
  'https://drive.usercontent.google.com/download?id=1_gvwQKbML8RzZhCZ_3gekoRMrb8BfMg-&export=download&confirm=t'

actual_bytes="$(stat -c %s "$destination")"
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
  echo "Unexpected dataset size: $actual_bytes bytes (expected $expected_bytes)" >&2
  exit 1
fi
echo "Downloaded $destination ($actual_bytes bytes)"

