from dataclasses import dataclass


@dataclass(frozen=True)
class MultimodalDatasetSpec:
    key: str
    name: str
    hsi_bands: int
    aux_modality: str
    aux_bands: int
    num_classes: int
    spatial_size: tuple[int, int]
    priority: str
    protocol_note: str
    risk_note: str = ""
    accepted_hsi_bands: tuple[int, ...] = ()
    required_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.accepted_hsi_bands:
            object.__setattr__(self, "accepted_hsi_bands", (self.hsi_bands,))


DATASET_SPECS: dict[str, MultimodalDatasetSpec] = {
    "houston2013_hs_lidar": MultimodalDatasetSpec(
        key="houston2013_hs_lidar",
        name="Houston2013-HS-LiDAR",
        hsi_bands=144,
        aux_modality="LiDAR",
        aux_bands=1,
        num_classes=15,
        spatial_size=(349, 1905),
        priority="p0",
        protocol_note="Current supported baseline; use fixed official ROI split first.",
        required_files=(
            "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat",
            "2013_IEEE_GRSS_DF_Contest_LiDAR.mat",
            "GRSS2013.mat",
            "2013_IEEE_GRSS_DF_Contest_Samples_TR.mat",
        ),
    ),
    "trento": MultimodalDatasetSpec(
        key="trento",
        name="Trento",
        hsi_bands=63,
        aux_modality="LiDAR/DSM",
        aux_bands=1,
        num_classes=6,
        spatial_size=(600, 166),
        priority="p0",
        protocol_note="Small standard HS-LiDAR benchmark; best first new dataset for loader migration.",
        required_files=(
            "HSI cube",
            "LiDAR/DSM raster",
            "ground-truth label map",
            "train/test masks or published split protocol",
        ),
    ),
    "muufl": MultimodalDatasetSpec(
        key="muufl",
        name="MUUFL Gulfport",
        hsi_bands=64,
        accepted_hsi_bands=(64, 72),
        aux_modality="LiDAR",
        aux_bands=2,
        num_classes=11,
        spatial_size=(325, 220),
        priority="p1",
        protocol_note="Challenging, imbalanced benchmark; confirm 64-band or 72-band convention before comparing.",
        risk_note="Class imbalance is severe; per-class AA/Kappa and split notes are mandatory.",
        required_files=(
            "HSI cube, preferably both raw 72-band and denoised 64-band if available",
            "two LiDAR rasters or processed LiDAR stack",
            "ground-truth label map",
            "train/test masks or exact paper split",
        ),
    ),
    "augsburg": MultimodalDatasetSpec(
        key="augsburg",
        name="Augsburg",
        hsi_bands=101,
        aux_modality="LiDAR",
        aux_bands=1,
        num_classes=6,
        spatial_size=(1364, 1636),
        priority="defer",
        protocol_note="Use only if the reconstructed label protocol is available.",
        risk_note="Published CAMFNet result relies on reconstructed labels; raw labels may be inconsistent.",
    ),
    "houston2018": MultimodalDatasetSpec(
        key="houston2018",
        name="Houston2018",
        hsi_bands=48,
        aux_modality="LiDAR",
        aux_bands=1,
        num_classes=20,
        spatial_size=(601, 2385),
        priority="defer",
        protocol_note="Ambitious large-scale stress test after Trento/MUUFL are stable.",
        risk_note="Extreme class imbalance and larger image size will slow the current patch pipeline.",
    ),
}


def recommended_dataset_sequence(
    include_deferred: bool = False,
) -> list[MultimodalDatasetSpec]:
    priority_order = {"p0": 0, "p1": 1, "p2": 2, "defer": 99}
    specs = [
        spec
        for spec in DATASET_SPECS.values()
        if include_deferred or spec.priority != "defer"
    ]
    return sorted(specs, key=lambda spec: (priority_order[spec.priority], spec.name))
