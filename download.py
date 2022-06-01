from pathlib import Path
import gdown

url = 'https://drive.google.com/uc?id=13NFG_H0aQXuuwq01WL03RJX28mZTeRpn'
path = Path("./data")
path.mkdir(exist_ok=True, parents=True)
gdown.download(url, "./data/data.zip")