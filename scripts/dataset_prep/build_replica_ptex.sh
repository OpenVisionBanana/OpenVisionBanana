#!/usr/bin/env bash
# Build the fixed-pose renderer against the official Replica SDK.
set -euo pipefail
destination=${1:?Usage: bash scripts/dataset_prep/build_replica_ptex.sh /writable/build-directory}
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$destination"
destination=$(cd -- "$destination" && pwd)
if [[ ! -d "$destination/Replica-Dataset/.git" ]]; then
  git clone --no-checkout https://github.com/facebookresearch/Replica-Dataset.git "$destination/Replica-Dataset"
fi
git -C "$destination/Replica-Dataset" checkout --detach 75fd55679ee189325d3593e838a762dd9163ee57
git -C "$destination/Replica-Dataset" submodule update --init --depth=1 3rdparty/Pangolin
cp "$repository/dataset_prep/replica/ptex_fixed_poses.cpp" "$destination/Replica-Dataset/ReplicaSDK/src/render.cpp"
# Repair the pinned SDK's headless EGL device selection in this build tree.
sed -i '/if (cudaDevNumber == cudaDevice) {/a\        foundCudaDev = true;' "$destination/Replica-Dataset/ReplicaSDK/ptex/EGL.cpp"
cmake -S "$destination/Replica-Dataset/3rdparty/Pangolin" -B "$destination/pangolin" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$destination/install" \
  -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} -include cstdint -include cstddef"
cmake --build "$destination/pangolin" --parallel "${BUILD_WORKERS:-4}"
cmake --install "$destination/pangolin"
cmake -S "$destination/Replica-Dataset" -B "$destination/renderer" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$destination/install" \
  -DPTEX_SHADER_DIR="$destination/Replica-Dataset/ReplicaSDK/shaders" \
  -DCMAKE_BUILD_RPATH="$destination/install/lib" -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} -include cstdint -include cstddef"
cmake --build "$destination/renderer" --target ReplicaRenderer --parallel "${BUILD_WORKERS:-4}"
printf 'Renderer: %s\n' "$destination/renderer/ReplicaSDK/ReplicaRenderer"
