# Controlled-Corruption Reliability Curves

This figure uses the three-seed Houston2013 priority summary to visualize how compact OA, auxiliary reliability, and auxiliary fusion weight change as the available LiDAR branch is progressively corrupted.

## Main Readout

- Under the selected multi-degradation p=0.25 setting, aux reliability changes by +0.023 from low to high noise and by -0.296 from 25% to 50% occlusion.
- Downsample-4 compact OA improves from 78.37% for the compact baseline to 84.15% with multi-degradation supervision.
- The fusion weight curves are diagnostic rather than a calibration guarantee: they expose whether the reliability branch changes behavior under controlled corruption, but they should not be interpreted as physical sensor-quality measurements.

## Generated Files

- `brmnet_controlled_corruption_reliability_curves.csv`: tidy source data.
- `brmnet_controlled_corruption_reliability_curves.pdf/.png`: repository figure.
- `fig_controlled_corruption_reliability_curves.pdf/.png`: paper figure copy.
