# Held-out depth protocol selection

`heldout_nyu_train32_v1.jsonl` freezes 32 NYUv2 training samples disjoint from
the 654-image test benchmark and from OVB training. They choose, once per
model/prompt/runtime/output representation, whether the held-out affine lane
uses metric-depth or disparity space. The selected family is then frozen for
every evaluation dataset. This indoor calibration is GT-assisted and must not
be described as a direct or GT-free metric.
