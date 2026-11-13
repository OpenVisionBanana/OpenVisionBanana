# Adds real native-COCO masks to the Objects365 recovery helper.
. "$ROOT/scripts/dataset_prep/coconut_xlarge_objects365_recovery.sh"
stage_coconut_xlarge_with_coco_and_objects365_recovery() {
  stage_coconut_xlarge_with_objects365_recovery
  # A disabled source needs no mask recovery (the delegate above already skipped).
  is_true "$COCONUT_XL_ENABLE" || return 0
  if [[ ",$COCONUT_XL_GROUPS," != *,coco,* ]]; then return; fi
  local assets source_root b_tar b_dir staged_root marker
  staged_root="$SEMSEG_STAGE/coconut_xlarge/coco"
  if [[ -n "$COCONUT_XL_ROOT" && "$COCONUT_XL_ROOT" != "$staged_root" ]]; then
    # The tree is used in place (never written to): it must already carry the
    # native COCONut-B masks, as the release downloader's materialized tree does.
    [[ -f "$COCONUT_XL_ROOT/panoptic_train2017/000000000009.png" ]] && return 0
    echo "ERROR: COCONut-XL tree used in place lacks native COCO masks: $COCONUT_XL_ROOT/panoptic_train2017" >&2
    exit 1
  fi
  if ! assets=$(resolve_coconut_xlarge_assets_root "$COCONUT_XL_SOURCE_ROOT"); then return; fi
  source_root=$(dirname -- "$assets")/_sources
  b_tar=$source_root/coconut_b/coconut_b.tar
  b_dir=$source_root/coconut_b/coconut_b
  if [[ ! -f "$b_tar" ]] && [[ -z $(find "$b_dir" -maxdepth 1 -name '*.png' -print -quit 2>/dev/null) ]]; then
    echo "ERROR: native COCONut-B masks requested but missing: $b_tar (and no extracted masks in $b_dir)" >&2
    exit 1
  fi
  marker="$staged_root/.ovb-coconut-b-masks-v1.complete"
  if [[ ! -f "$marker" || ! -f "$staged_root/panoptic_train2017/000000000009.png" ]]; then
    find "$staged_root/panoptic_train2017" -maxdepth 1 -type l -name '0*.png' -delete
    echo "Recovering native COCONut-B panoptic masks into $staged_root/panoptic_train2017"
    if [[ -f "$b_tar" ]]; then
      tar --strip-components=1 -xf "$b_tar" -C "$staged_root/panoptic_train2017"
    else
      cp -al "$b_dir/." "$staged_root/panoptic_train2017/" 2>/dev/null \
        || cp -a "$b_dir/." "$staged_root/panoptic_train2017/"
    fi
    [[ -f "$staged_root/panoptic_train2017/000000000009.png" ]] || { echo "ERROR: native COCONut-B mask recovery failed" >&2; exit 1; }
    touch "$marker"
  fi
}
