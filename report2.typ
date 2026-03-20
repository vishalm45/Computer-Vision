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
Segmenting objects cleanly from cluttered scenes is a core problem in computer vision and sits behind many real-world systems, from medical imaging to robotics @szeliski_computer_2010. Here, the target is a coloured parachute moving through a dark sky, with 51 RGB frames and a matching set of binary ground-truth masks. The Dice Similarity Coefficient is also used to measure accuracy, which is a standard overlap score for segmentation @noauthor_dice_nodate.

= Methodology
== Pipeline
For each RGB image, the algorithm follows four steps
- Preprocess the image and capture a colour representation
- Build a parachute likelihood image from the colour cues
- Perform region growing from an automatically chosen seed region
- Apply morphological post-processing and connected component analysis to keep a single final object.

This strucutre fits within the traditional region-based segmentation family. Low level cues first mark likely foreground pixels, then morphology and connected components regularise the shape and remove clutter @parvati_image_2008. Automatic operations such as adaptive thresholding and seed selection are used wherever possible so that the method depends only weakly on the specific brightness of each frame @mangal_robust_2024.

== Preprocessing and colour representation
Each input frame is read as an RGB image, converted to double precision in the range [0,1], and lightly smoothed with a Gaussian filter. The smoothing supresses fine-scale noise without removing the parachute edges, which is important before thresholding or region growing @yamaguchi_edge-preserving_2023.

Rather than converting to CIE LAB, the implementation stays in RGB, but constructs a colourfulness measure that is tailored to this dataset. R, G, and B represent the red, green and blue channels. The parachute is strong in red and green, but weak in blue, whereas the sky remains grey and blueish. To utilise this, a colour score is defined as #align(center, $ C = max(R,G) - B + 0.6 |R - G|$)

The first term highlights pixels that are brighter in read or green than in blue, and the second term favours locations where the red-green difference is large, which is true on the stripes of the parachute. The combined map is then scaled into [0,1] using min-max normalisation. Using colour distances in this way matches the idea that perceptual differences in colour can help separate objects from the background @sharma_ciede2000_2005.

== Spatial prior and intiial thresholding
Inspection of the sequence shows that the parachute remains roughly in the middle of the frame. The parachute is never on the very left or right, and never reaches the top and bottom borders. This can be encoded as a spatial prior by defining a rectangular region of interest (ROI) covering the central band of the image both horizontally and vertically. Pixels that are ouside this ROI are set to zero in the colour score map so that later stages ignore these areas.

Within this ROI an initial binary mask is produced using MATLAB's adaptive thresholding function followed by binarisation. Adaptive thresholding estimates a threshold from local statistics rather than from the global histogram, making it more robust to the strong illumination gradient caused by the light beam @helmy_understanding_2023. After thresholding, very small connected components are discarded with an area filter to remove isolated noise blobs and small specular highlights.

== Seed selection and geodesic region growing
The threholded mask typically contains several surviving blobs. Some of these correspond to parts of the parachute, but others can be small pieces of the cave or light shaft. To pick a good starting point for region growing, connected component analysis is applied and a handful of shape features are computed for each component, such as area, eccentricity, perimeter and circularity $4 pi A "/" P^2$ @doube_multithreaded_nodate. A simple score then favours components that have an area in the expected range for the parachute, are compact, which is high circularity and those that are not extremely elongated, with a low eccentricity. The component with the highest score is used as a seed. If this seed is extremely small, it is dilated to give region growing a more stable starting set.

To obtain an appropriate object region, the method uses geodesic region growing on the colour score image. A geodesic distance map is computed using MATLAB's fast marching-based tools, where pixels similar in colour to the seed have low cost and dissimilar pixels have high cost. Thresholding this map produces a region of pixels that can be reached from the seed via paths with small accumulated colour differences. This is consistent with region-based segmentation schemes where similarity is measured along spatial paths rather than per-pixel only @parvati_image_2008.

== Morphological refinement and final object selection
The grown region is binarised and refined with standard morphological operations. Small components are removed using area opening, internal gaps are filled with hole filling, and a mild opening operation smooths irregular boundaries @parvati_image_2008. These steps are important to suppress stray pixels that region growing may have picked up while still keeping the overall canopy shape intact.

Finally, connected-component analysis is appied once more. For each remaining component, a compactness-based score similar to the seed selected step is computed, combining area, circularity and eccentricity. Only the highest scoring component is kept as the final parachute mask for that frame, reflecting the prior knowledge that only one parachute is present. If needed, the mask is resized with nearest-neighbour interpolation so that it matches the resolution of the ground-truth mask before evaluation @taha_metrics_2015.

