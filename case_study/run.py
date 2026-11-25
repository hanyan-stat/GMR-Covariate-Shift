"""Rebuild the GAMA sample, run the case-study analyses and format their tables."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

METHODS = ("Target reference", "OEL-DW", "Plug-Sieve", "Plug-DRF", "LDML", "DRW")
CATALOGUES = {
    "TilingCatv46.fits": (
        "EqInputCat/v46",
        "c9dc8e098a76ede238f9d0d6b2ccd953ed3681ea86b79b04a741e8915942e67b",
    ),
    "ExternalSpecAllv07.fits": (
        "ExternalSpec/v07",
        "287e6bc5d467a8eadb99e2c2c8838664097b4bde937ac808ccfd5b881e3230e6",
    ),
}


def read_catalogue(name, directory):
    """Read verified FITS bytes from the given directory or the public archive."""
    from astropy.io import fits

    section, checksum = CATALOGUES[name]
    if directory is None:
        url = f"https://www.gama-survey.org/dr4/data/cat/{section}/{name}"
        with urlopen(url, timeout=180) as stream:
            data = stream.read()
    else:
        data = (Path(directory) / name).read_bytes()
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"Catalogue checksum differs for {name}")
    with fits.open(BytesIO(data), memmap=False) as catalog:
        return catalog[1].data.copy()


def catalogue_sample(directory=None):
    """Return the source and target arrays in CATAID order, with target labels separate."""
    import numpy as np

    t = read_catalogue("TilingCatv46.fits", directory)
    s = read_catalogue("ExternalSpecAllv07.fits", directory)
    s = s[(s["SURVEY"] == "SDSS") & (s["NQ"] >= 3) & (s["IS_SBEST"] == 1) & (s["CATAID"] > 0)]
    s = s[np.argsort(s["CATAID"])]
    if len(np.unique(t["CATAID"])) != len(t) or len(np.unique(s["CATAID"])) != len(s):
        raise ValueError("The catalogue join requires unique galaxy IDs")
    source = np.isin(t["CATAID"], s["CATAID"])
    keep = (t["SURVEY_CLASS"] >= 4) & ((t["TARGET_FLAGS"] & 1) != 0)
    for column, low, high in [
        ("R_PETRO", 14, 17.77),
        ("R_SB", 15, 24.5),
        ("FIBERMAG_R", 16.6, 22.6),
    ]:
        keep &= (t[column] >= low) & (t[column] <= high)
    keep &= t["SG_SEP"] >= 0.25
    for band in "UGRIZ":
        mag = t[f"{band}_MODEL"]
        keep &= np.isfinite(mag) & (mag > 5) & (mag < 35)
    cell = np.floor(t["RA"]).astype(int) * 100 + np.floor(t["DEC"]).astype(int) + 20
    cells, count = np.unique(cell[keep & source], return_counts=True)
    keep &= np.isin(cell, cells[count >= 20])
    keep &= source | (t["NQ"] >= 3)
    t, source = t[keep], source[keep]
    order = np.argsort(t["CATAID"])
    t, source = t[order], source[order]
    field = np.where(t["RA"] < 150, 0, np.where(t["RA"] < 200, 1, 2))
    x = np.column_stack(
        [
            t["R_PETRO"],
            t["U_MODEL"] - t["G_MODEL"],
            t["G_MODEL"] - t["R_MODEL"],
            t["R_MODEL"] - t["I_MODEL"],
            t["I_MODEL"] - t["Z_MODEL"],
            t["FIBERMAG_R"],
            t["R_SB"],
            t["SG_SEP"],
            t["NEIGHBOUR_CLASS"],
            t["RA"] - np.array([135.0, 180.0, 217.5])[field],
            t["DEC"],
            field == 1,
            field == 2,
        ]
    ).astype(float)
    position = np.column_stack([t["RA"], t["DEC"]]).astype(float)
    idx = np.searchsorted(s["CATAID"], t["CATAID"][source])
    yp, yq = s["Z"][idx].astype(float), t["Z"][~source].astype(float)
    if (len(yp), len(yq)) != (13085, 1935) or not all(np.isfinite(v).all() for v in (x, yp, yq)):
        raise ValueError("Unexpected GAMA analysis sample")
    return (x[source], yp, x[~source], position[source], position[~source]), yq


def compare(study, target_redshift):
    from . import pointwise_inference as pointwise

    ours, scores = pointwise.oel(study)
    plugs, projected = pointwise.plugins(study)
    rows = {"OEL-DW": ours, **plugs, "LDML": pointwise.ldml(study), "DRW": pointwise.drw(study)}
    # Target labels first enter here, after the five transfer procedures are finished.
    target_error = abs(study.hq - target_redshift) / (1.0 + target_redshift)
    benchmark, learners, ref_scores = pointwise.references(study, target_error)
    rows["Target reference"] = benchmark
    return dict(
        main={name: rows[name] for name in METHODS},
        projected_sieve=projected,
        reference_learners=learners,
        spatial={
            name: pointwise.spatial_scores(study, component)
            for name, component in [("OEL-DW", scores), ("Target reference", ref_scores)]
        },
    )


def tables(result):
    """Print the three case-study tables in the units of the paper as TSV."""

    def display(value):
        return format(100 * value, "#.3g").rstrip(".")

    def interval(row):
        return [display(row["estimate"]), f"[{display(row['lower'])}, {display(row['upper'])}]"]

    def show(values):
        print("\t".join(map(str, values)))

    print("Main paper: galaxy prediction errors (x100)")
    show(["Method", "theta1", "95% CI", "CI length", "theta2", "95% CI", "CI length"])
    for method in METHODS:
        values = [method]
        for row in result["main"][method]:
            values.extend([*interval(row), display(row["ci_length"])])
        show(values)
    print("\nSupplement: reference learners and projected Plug-Sieve (x100)")
    show(["Parameter", "Method", "Estimate", "95% CI"])
    for j in range(2):
        rows = {"Target reference": result["main"]["Target reference"][j]}
        if j == 1:
            rows.update(
                {
                    (
                        "Target reference (forest average)"
                        if name == "Ensemble"
                        else f"Target reference ({name})"
                    ): result["reference_learners"][name]
                    for name in ("DRF", "ExtraTrees", "Ensemble")
                }
            )
        rows.update(
            {
                "OEL-DW": result["main"]["OEL-DW"][j],
                "Projected Plug-Sieve": result["projected_sieve"][j],
            }
        )
        for name, row in rows.items():
            show([f"theta{j+1}", name, *interval(row)])
    print("\nSupplement: spatial standard errors (x100) and variance shares (%)")
    show(
        [
            "Parameter",
            "Method",
            "Independent",
            "1-degree cells",
            "2-degree cells",
            "Target share",
            "Top five",
        ]
    )
    for j in range(2):
        for name, rows in result["spatial"].items():
            show(
                [
                    f"theta{j+1}",
                    name,
                    *(
                        display(rows[j][k])
                        for k in ("independent", "sky_1", "sky_2", "target_share", "top_five")
                    ),
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("pointwise", "Fit Table 4 and both supplementary tables"),
        ("curve", "Fit all 17 quantiles and calibrate the joint EL band"),
        ("full-source", "Refit the supplementary full-source sensitivity band"),
    ]:
        group = commands.add_parser(name, help=help_text)
        group.add_argument(
            "--raw", type=Path, help="Directory containing the two original FITS files"
        )
    render = commands.add_parser("tables", help="Format saved pointwise JSON without fitting")
    render.add_argument("json_file", type=Path)
    args = parser.parse_args()
    if args.command == "tables":
        tables(json.loads(args.json_file.read_text(encoding="utf-8")))
        return
    from simulation.nuisance import initialize
    from .pointwise_inference import fix_predictor

    initialize()
    sample, target_redshift = catalogue_sample(args.raw)
    study = fix_predictor(sample)
    if args.command == "pointwise":
        result = compare(study, target_redshift)
    else:
        from .quantile_band import estimate_band, full_source_band, evaluate

        result = estimate_band(study) if args.command == "curve" else full_source_band(study)
        target_error = abs(study.hq - target_redshift) / (1.0 + target_redshift)
        evaluate(result, target_error)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
