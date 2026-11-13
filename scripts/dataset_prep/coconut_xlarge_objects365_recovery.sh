# Requires the staging launcher's helpers and globals.
stage_coconut_xlarge_with_objects365_recovery() {
  if [[ ",$COCONUT_XL_GROUPS," != *,objects365,* ]]; then stage_coconut_xlarge_if_needed; return; fi
  local assets source_root large_tar xl_tar archives_dir stage_base staged_root marker image_marker mask_marker archive
  if ! assets=$(resolve_coconut_xlarge_assets_root "$COCONUT_XL_SOURCE_ROOT"); then stage_coconut_xlarge_if_needed; return; fi
  source_root=$(dirname -- "$assets")/_sources
  large_tar=$source_root/coconut_large/panoptic_object365.tar; xl_tar=$source_root/coconut_xlarge/coconuts_xlarge.tar; archives_dir=$source_root/objects365_archives
  if [[ -f "$assets/train2017/objects365_v2_00940000.jpg" && -f "$assets/panoptic_train2017/objects365_v2_00940000.png" ]]; then stage_coconut_xlarge_if_needed; return; fi
  if [[ ! -f "$large_tar" || ! -f "$xl_tar" || ! -d "$archives_dir" ]]; then echo "WARNING: Objects365 recovery archives are unavailable; using normal COCONut staging." >&2; stage_coconut_xlarge_if_needed; return; fi
  stage_base="$SEMSEG_STAGE/coconut_xlarge"; staged_root="$stage_base/coco"; marker="$stage_base/.ovb-objects365-recovery-v1.complete"; image_marker="$stage_base/.ovb-objects365-images.complete"; mask_marker="$stage_base/.ovb-objects365-masks.complete"
  if is_true "$FORCE"; then rm -rf "$stage_base"; fi
  mkdir -p "$staged_root/train2017" "$staged_root/panoptic_train2017"; ln -sfn "$assets/annotations" "$staged_root/annotations"
  if [[ ! -f "$marker" ]]; then cp -al "$assets/train2017/." "$staged_root/train2017/"; fi
  if [[ ! -f "$image_marker" ]]; then
    find "$staged_root/train2017" -maxdepth 1 -type l -name 'objects365_v2_*.jpg' -delete
    echo "Recovering Objects365 RGB patches into $staged_root/train2017"
    while IFS= read -r archive; do tar --skip-old-files --strip-components=1 -xzf "$archive" -C "$staged_root/train2017"; done < <(find "$archives_dir" -maxdepth 1 -type f -name 'patch*.tar.gz' -printf '%p\n' | sort)
    touch "$image_marker"
  fi
  if [[ ! -f "$mask_marker" ]]; then
    archive="$assets/panoptic_train2017.tar"; [[ -f "$archive" ]] && tar -xf "$archive" -C "$staged_root"
    mkdir -p "$staged_root/panoptic_train2017"; find "$staged_root/panoptic_train2017" -maxdepth 1 -type l -name 'objects365_v2_*.png' -delete
    echo "Recovering Objects365 panoptic masks into $staged_root/panoptic_train2017"
    tar --wildcards --strip-components=2 -xf "$xl_tar" -C "$staged_root/panoptic_train2017" 'coconuts_xlarge/panseg/*.png'
    tar --skip-old-files --strip-components=1 -xf "$large_tar" -C "$staged_root/panoptic_train2017"
    [[ -f "$staged_root/panoptic_train2017/objects365_v2_00940000.png" ]] || { echo "ERROR: Objects365 mask recovery failed" >&2; exit 1; }
    touch "$mask_marker"
  fi
  touch "$marker"; COCONUT_XL_ROOT=$staged_root
}
