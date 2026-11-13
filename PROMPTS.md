# Quick-start prompts

One prompt per task. Use any of them as `prompt` in the pipeline call of the
[README quick start](README.md#-quick-start). The call stays the same for every
task except text-to-image generation, which takes no input image.

<img src="assets/labels/semseg.png" width="255" alt="Semantic segmentation">

```python
prompt = (
    "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of "
    'the input image, using this color mapping: {"road": [8,4,249], "sidewalk": [255,8,232], '
    '"vegetation": [10,255,2], "sky": [2,204,254], "person": [255,0,13], "car": [255,249,23]} '
    "Color each pixel of a listed class with its color and leave everything else black. "
    "Output only the segmentation visualization image."
)
```

<img src="assets/labels/refseg.png" width="257" alt="Referring segmentation">

```python
prompt = (
    "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this "
    'description: "table in the room". Paint the matching region red [220, 30, 30] and '
    "leave everything else black [0, 0, 0]. Output only the segmentation mask image."
)
```

<img src="assets/labels/instanceseg.png" width="248" alt="Instance segmentation">

```python
prompt = (
    "[TASK: INSTANCE_SEGMENTATION] This image shows segmentation masks for car from the input image. "
    "The background is set to black color [0, 0, 0]. Each car instance is represented by a solid mask, "
    "and a different color is used for each mask. Instances are numbered by where their mask first "
    "appears in a left-to-right scan of the image (ties broken top to bottom). The 1st car is blue, "
    "the 2nd is lime, the 3rd is red, the 4th is sky blue, the 5th is dark green, the 6th is pink, "
    "the 7th is apricot, the 8th is mint, the 9th is wine, the 10th is magenta, the 11th is navy, "
    "the 12th is green, the 13th is yellow, the 14th is brown, the 15th is white, the 16th is olive, "
    "the 17th is lavender, the 18th is teal, the 19th is orange, the 20th is hot pink, the 21st is plum, "
    "the 22nd is gray, the 23rd is purple, the 24th is beige, the 25th is steel blue, the 26th is cyan, "
    "the 27th is royal blue, the 28th is sea green, the 29th is rust, the 30th is mustard, "
    "the 31st is mauve, the 32nd is maroon."
)
```

<img src="assets/labels/depth.png" width="194" alt="Depth estimation">

```python
prompt = (
    "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
    "for every input pixel at the same pixel location, with closer points brighter and farther points "
    "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
    "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
    "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
    "legends, borders, or objects; return only the registered depth map."
)
```

<img src="assets/labels/normals.png" width="282" alt="Surface normal estimation">

```python
prompt = (
    "Create a dense camera-space unit surface-normal map registered to the input. Use +x image-right, "
    "+y image-up, +z toward the camera and encode (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. "
    "Preserve planes, curved surfaces, thin structures, and sharp orientation boundaries; "
    "return only the RGB normal map."
)
```

<img src="assets/labels/imgedit.png" width="161" alt="Image editing">

```python
prompt = (
    "[TASK: IMAGE_EDITING] Change the beach and ocean environment in the picture to a snowy mountain."
)
```

<img src="assets/labels/generation.png" width="271" alt="Text-to-image generation">

Call the pipeline without `image` and give it a size:

```python
prompt = "an empty tram at a cobblestone stop on a rainy night"
output = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

## Adapting the prompts

The model was trained on many phrasings of each task, so the prompts above can
be adapted.

- **Semantic segmentation:** list any classes with an RGB colour each. Distinct, saturated colours work best. Pixels outside the listed classes come out black.
- **Referring segmentation:** replace the description and the paint colour.
- **Instance segmentation:** prompt one class at a time and replace every `car` in the prompt with any noun phrase. The left-to-right colour ordering keeps the assigned colours well separated. The model may not follow the ordering exactly, which does not matter, because each instance only needs its own distinct colour.
- **Depth and surface normals:** use the prompts as they are. Depth is relative inverse depth, with closer points brighter. Normals follow the encoding stated in the prompt.
- **Image editing:** keep the `[TASK: IMAGE_EDITING]` tag and write the edit as a plain instruction.
- **Generation:** a plain description without a tag and without an input image.

The exact prompt templates used for the benchmarks are built by the evaluation
front ends described in [EVALUATION.md](EVALUATION.md).
