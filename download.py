import os
import gdown
url = 'https://drive.google.com/uc?id=13NFG_H0aQXuuwq01WL03RJX28mZTeRpn'
os.mkdir("data")
gdown.download(url, "./data/data.zip")
