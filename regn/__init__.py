from pathlib import Path
regn_path = Path(__file__).parent.parent
from urllib.request import urlretrieve

def download_data():

    data_folder = regn_path / "data"
    data_folder.mkdir(exist_ok=True)
    datasets = ["training_data_gmi_small.nc",
                "training_data_gmi_large.nc",
                "test_data_gmi_small.nc",
                "validation_data_gmi_small.nc",
                "training_data_mhs_small.nc",
                "training_data_mhs_large.nc",
                "test_data_mhs_small.nc",
                "validation_data_mhs_small.nc"]
    for file in datasets:
        file_path = data_folder / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/regn/{file}"
            urlretrieve(url, file_path)

    model_folder = regn_path / "models"
    model_folder.mkdir(exist_ok=True)
    models = ["qrnn_gmi.pt",
              "qrnn_mhs.pt"]
    for file in models:
        file_path = model_folder / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/regn/{file}"
            urlretrieve(url, file_path)

download_data()
