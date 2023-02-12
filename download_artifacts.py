import os
import gdown
import zipfile
import shutil

drive = {
    "dataset": {
        "url": "https://drive.google.com/file/d/19lAC-ITVrdbjyYBkjBU5p6-QJFnFyKGK/view?usp=sharing",
        "id": "19lAC-ITVrdbjyYBkjBU5p6-QJFnFyKGK",
        "file": "b2w/dataset.zip",
        "path": "b2w/dataset",
    },
    "models": {
        "url": "https://drive.google.com/file/d/1PxiMiaVB9TkH4Nh7j2Ofe-TXrkCb1xH9/view?usp=sharing",
        "id": "1PxiMiaVB9TkH4Nh7j2Ofe-TXrkCb1xH9",
        "file": "b2w/models.zip",
        "path": "b2w/models",
    },
    "data_with_embeddings": {
        "url": "https://drive.google.com/file/d/1R-63yWHIqFIe4lSKXToK20i6Vt6i6hUG/view?usp=sharing",
        "id": "1R-63yWHIqFIe4lSKXToK20i6Vt6i6hUG",
        "file": "b2w/data_with_embeddings.zip",
        "path": "b2w/data_with_embeddings",
    },
}


def GDDownload(folder):
    if folder in drive.keys():
        os.makedirs(os.path.dirname(drive[folder]["path"]), exist_ok=True)

        gdown.download(drive[folder]["url"], drive[folder]["file"], fuzzy=True)

        with zipfile.ZipFile(drive[folder]["file"], "r") as zip_ref:
            zip_ref.extractall(drive[folder]["path"])

        os.remove(drive[folder]["file"])
    else:
        print(
            "Folder doesnt exist! Please enter one of the options: ", list(drive.keys())
        )


if __name__ == "__main__":
    shutil.rmtree("b2w")
    drive_files = ["dataset", "models", "data_with_embeddings"]
    for f in drive_files:
        GDDownload(f)
