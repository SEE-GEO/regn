from pathlib import Path
regn_path = Path(__file__).parent.parent
from urllib.request import urlretrieve

def download_data():

    datasets = ["training_data_gmi_small.nc",
                "training_data_gmi_large.nc",
                "test_data_gmi_small.nc",
                "validation_data_gmi_small.nc"]

    for file in datasets:
        file_path = regn_path / "data" / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/regn/{file}"
            urlretrieve(url, file_path)

    models = ["qrnn_gmi.pt"]
    for file in models:
        file_path = regn_path / "models" / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/regn/{file}"
            urlretrieve(url, file_path)

download_data()
