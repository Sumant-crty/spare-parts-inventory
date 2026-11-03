"""Download the car-parts dataset (Monash Time Series Forecasting Repository, CC BY 4.0) into data/.

Usage: python scripts/download_data.py
"""

import io
import pathlib
import urllib.request
import zipfile

URL = "https://zenodo.org/records/4656022/files/car_parts_dataset_with_missing_values.zip?download=1"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    target = DATA / "car_parts_dataset_with_missing_values.tsf"
    if target.exists():
        print("already downloaded:", target)
        return
    # Zenodo rejects generic browser user agents, so identify the client honestly
    headers = {"User-Agent": "spare-parts-inventory/0.1 (+https://github.com/Sumant-crty/spare-parts-inventory)"}
    with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA)
    print("saved", target)


if __name__ == "__main__":
    main()
