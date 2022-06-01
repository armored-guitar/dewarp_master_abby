import zipfile
import os

with zipfile.ZipFile("./data/data.zip", 'r') as zip_ref:
    zip_ref.extractall("./data/")

import os
# os.remove("./data/data.zip")
