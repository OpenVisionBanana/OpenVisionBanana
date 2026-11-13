#!/usr/bin/env bash
# Download the PACO-LVIS DatasetNinja archive without extracting it, to
# instanceseg/paco-lvis-DatasetNinja.tar, where staging reads it
# (instanceseg/paco_lvis/ is reserved for an already extracted tree).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/datasets_archives/instanceseg}"
ARCHIVE_PATH="$ARCHIVE_DIR/paco-lvis-DatasetNinja.tar"
SOURCE_URL="https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjUxMl9QQUNPLUxWSVMvcGFjby1sdmlzLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogImtiRG5WY2trQlNZbDhJMWpRRWlZU09JZUdyT0VWYkRIL2E3cXpnMlpwV3c9In0=?response-content-disposition=attachment%3B%20filename%3D%22paco-lvis-DatasetNinja.tar%22"

if [[ -s "$ARCHIVE_PATH" ]]; then
  echo "Using existing $ARCHIVE_PATH"
  exit 0
fi
mkdir -p "$ARCHIVE_DIR"

# The final name appears only after a complete download; reruns resume the .partial.
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --continue-at - --output "$ARCHIVE_PATH.partial" "$SOURCE_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --continue --output-document="$ARCHIVE_PATH.partial" "$SOURCE_URL"
else
  echo "This script requires curl or wget." >&2
  exit 1
fi
mv -f "$ARCHIVE_PATH.partial" "$ARCHIVE_PATH"

echo "Downloaded PACO-LVIS archive to: $ARCHIVE_PATH"
