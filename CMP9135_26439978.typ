#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [CMP9135 Computer Vision],
  authors: (
    (
      name: "Vishal Maisuria",
      department: [MSc Artificial Intelligence],
      location: [University of Lincoln],
      email: "26439978@students.lincoln.ac.uk"
    ),
  ),
  bibliography: bibliography("refs.bib", style: "university-of-lincoln-harvard.csl", full: true),
  figure-supplement: [Fig.],
) 

= Introduction

A central problem in computer vision is the task of segmenting objects from complex visual scenes, which underpins applications such as medical imaging *REFERENCE*. Here, I segment a parachute from a set of 51 RGB images, each paired with a ground-truth mask, using only classical computer vision techniques. 

Performance is assessed using the Dice Similarity Coefficient (DSC), which is an overlap metric for segmentation quality *REFERENCE*.

= Methodology
== Pipeline
The implemented pipeline consists of four stages:
- Preprocessing and colour space transformation
- Extraction of luminance and colour based cues
- The fusion of cues
- Morphological refinement and analysis

All of these combined to produce a single parachute region. Each stage is designed to rely on automatic procedures such as Otsu thresholding and contrast-limited adaptive histogram equalisation (CLAHE). This helps the approach to generalise to unseen images *REFERENCE*.

The design follows a classic region-based segmentation approach, where intensity and colour statistics are first used to generate candidate foreground regions, which are then cleaned and regularised using morphology *REFERENCE*. This is to keep the model simple and interpretable.

== Preprocessing and colour representation
The RGB input images are first converted to the CIE LAB colour space, separating luminance L from chromaticity channels a and b. LAB is approximately perceptually uniform, and is used for colour-based segmentation because Euclidean distances in this space more closely reflect human colour differences than in RGB *REFERENCE*. For the parachute images, this separation is beneficial as the parachutes are brighter and more colourful than the background, allowing intensity and chroma to be treated as independent cues.

To improve the separability of foreground and background in the luminance channel, Contrast Limited Adaptive Histogram Equalisation (CLAHE) is applied to L before thresholding. CLAHE performs local histogram equalisation while clipping large peaks to avoid over-amplifying noise, making it particularly effective in images with varying lighting conditions *REFERENCE*. This step noticeably enhances the parachute structure in shadowed regions, without introducing excessive noise. To suppress small scale noise while keeping edges intact, a Guassian Blue is applied to the equalised luminance *REFERENCE*.

== Luminance and colour based foreground cues
After preprocessing, two complementary cues are extracted. Firstly, a luminance cue, obtained by applying Otsu's method to the equalised luminance channel. Otsu thresholding selects the threshold that maximises the between-class variance of foreground and background, effectively choosing an intensity level that best separates two dominant pixel classes under minimal assumption *REFERENCE*. Using Otsu avoids manually tuning thresholds, along with adapting automatically to images with different lighting conditions. 

The second cue relies on colour information. Based on LAB representation, a chroma magnitude image is computed as the Euclidean distance of each pixel's (a,b) vector from the neutral point (128,128) in the 8-bit encoding. The magnitude approximates how saturated each pixel colour is relative to grey *REFERENCE*. The parachute canopy has a higher saturation, so regions of high chroma are strong cadidates for parachute pixels. Otsu's thresholding is used on the chroma image to obtain a binary colourfulness mask, which ensures that the segmentation adapts to the exact palette of each image without fixing a particular chroma threshold.

By extracting these two cues, the pipeline captures both the brightness contrast and the distinctive colour of the parachute, while remaining data driven. During implementation, this dual-cue approach proved much more reliable than using luminance or colour alone, particularly in images where the parachute is shadowed or the background had bright parts.

== Cue fusion and fallbacks
The luminance and chroma masks are initially combined by a logical intersection, so that only pixels that are both bright and colourful are considered as foreground. This strategy reduces false positive from bright but grey regions, or from colourful noise in darker parts of the image. These AND style combinations of cues are useful when multiple independent features must be satisfied at the same time *REFERENCE*.

However, intersecting the two masks can be too strict in challenging cases where one cue is degraded, for example when the parachute is dark. To solve this, an adaptive fallback is implemented. If the combined mask contains fewer pixels than a fraction of the luminance mask, the fusion reverts to the union of the two cues instead. This prevents cases where a small misalignment between thresholds leads to an empty mask. 