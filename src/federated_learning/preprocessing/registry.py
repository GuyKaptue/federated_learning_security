"""Registry of the public real-world datasets used in the study (sources, checksums, licences).

Raw files are fetched on demand into data/raw/<key>/ and verified against the pinned SHA-256 below. They are
NOT redistributed with this repository. Licence statuses are recorded as found; `verified` means the licence was read
on the cited original source page on 2026-11-06 -- it does NOT extend to the third-party GitHub mirrors we download from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

COMMIT_SITES = "3103f98b92f8ee2ab25900a3f1ed31f096b17c23"
BASE_SITES = f"https://raw.githubusercontent.com/byte7/heart-disease-analysis-and-prediction/{COMMIT_SITES}/original_datasets/"


@dataclass(frozen=True)
class RawFile:
    filename: str
    url: str | None            # None = bundled with a library (no download)
    sha256: str | None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    alias: str                 # short name used in code and experiments
    title: str
    n_samples: int             # rows in the raw source file(s)
    n_features: int
    target: str
    positive_class: str
    source_page: str
    download: str
    license: str
    license_status: str
    citation: str
    usage: str
    notes: str = ""
    files: tuple[RawFile, ...] = field(default_factory=tuple)


SPECS: dict[str, DatasetSpec] = {s.key: s for s in (
    DatasetSpec(
        key="wisconsin_breast_cancer", alias="wdbc", title="Breast Cancer Wisconsin (Diagnostic)", n_samples=569, n_features=30,
        target="diagnosis (1 = benign, 0 = malignant in scikit-learn coding)", positive_class="benign (label 1)",
        source_page="https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic",
        download="bundled with scikit-learn (sklearn.datasets.load_breast_cancer); no download",
        license="CC BY 4.0", license_status="verified on the UCI dataset page (2026-11-06)",
        citation="Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B",
        usage="federated-learning validation with synthetic client partitions",
        files=(RawFile("wdbc_sklearn_arrays.sha256", None, "25e2eb7b7a8745b1bde5ba18bc4969983d2e2f403291ecd5bde4a29ef30c3859"),),
        notes="The checksum covers the float64 feature matrix followed by the int64 target vector as exposed by scikit-learn."),
    DatasetSpec(
        key="pima_diabetes", alias="pima", title="Pima Indians Diabetes Database", n_samples=768, n_features=8,
        target="onset of diabetes within 5 years (1 = positive)", positive_class="diabetes (label 1)",
        source_page="https://archive.ics.uci.edu/ (original owner: National Institute of Diabetes and Digestive and Kidney Diseases; Smith et al. 1988)",
        download="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv (third-party GitHub mirror)",
        license="not verified", license_status="no licence statement found on the pages we could read; mirror licence unknown",
        citation="Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proc. Symp. Computer Applications and Medical Care, 261-265 (reference NOT verified).",
        usage="federated-learning validation with synthetic client partitions",
        files=(RawFile("pima.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"),),
        notes="Zeros in glucose, blood pressure, skin thickness, insulin and BMI are physiologically impossible and treated as missing (imputed with the TRAIN median)."),
    DatasetSpec(
        key="cleveland_heart_disease", alias="heart", title="Heart Disease (Cleveland, processed 14-attribute table)", n_samples=303, n_features=13,
        target="binary target as provided by the mirror (target = 1 for 164 of 302 rows)", positive_class="label 1 = ABSENCE of disease (inferred, see notes)",
        source_page="https://archive.ics.uci.edu/dataset/45/heart-disease",
        download="https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv (third-party GitHub mirror)",
        license="CC BY 4.0 (UCI Heart Disease dataset)", license_status="verified on the UCI page for the original dataset; mirror licence not verified",
        citation="Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X",
        usage="federated-learning validation with synthetic client partitions",
        files=(RawFile("heart.csv", "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv", "e31e52eb5ee890c1a11a3baef3df95e8174718d5b94b0a8b77665ec3c07328ff"),),
        notes="303 raw rows; one exact duplicate row is removed (302 used). UCI states names and social security numbers were removed/replaced. Label semantics: the mirror has 164 rows with target=1, which equals the number of Cleveland patients WITHOUT disease in the UCI site file (164 of 303; 139 have disease), so target=1 appears to mean absence of disease. This was inferred from counts, not documented by the mirror. It does not affect binary-classification metrics, but 'positive class' here is not 'disease'."),
    DatasetSpec(
        key="uci_heart_disease_multisite", alias="heart_sites", title="Heart Disease, four collection sites (Cleveland, Hungary, Switzerland, VA Long Beach)", n_samples=920, n_features=10,
        target="angiographic disease (num > 0)", positive_class="disease (label 1)",
        source_page="https://archive.ics.uci.edu/dataset/45/heart-disease",
        download=BASE_SITES + "{cleveland14,hungarian14r,switzerland14,long_beach_va14}.csv (third-party GitHub mirror pinned to a commit)",
        license="CC BY 4.0 (UCI Heart Disease dataset)", license_status="verified on the UCI page for the original dataset; mirror licence not verified",
        citation="Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X",
        usage="natural site-based federated clients (one site = one client), within-site and leave-one-site-out evaluation",
        files=(RawFile("cleveland14.csv", BASE_SITES + "cleveland14.csv", "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"),
               RawFile("hungarian14r.csv", BASE_SITES + "hungarian14r.csv", "18b0496cd61717608e417d67c2ffacfedae87efce9ea68de9ef20fba66e8e0a1"),
               RawFile("switzerland14.csv", BASE_SITES + "switzerland14.csv", "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314"),
               RawFile("long_beach_va14.csv", BASE_SITES + "long_beach_va14.csv", "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f")),
        notes="Ten features kept (slope, ca, thal dropped: >= 50% missing at three of four sites). Cholesterol coded 0 for every Swiss patient is treated as missing."),
)}
ALIASES = {s.alias: s.key for s in SPECS.values()}


def get_spec(name: str) -> DatasetSpec:
    key = ALIASES.get(name, name)
    if key not in SPECS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    return SPECS[key]
