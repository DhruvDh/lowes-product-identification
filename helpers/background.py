import urllib.request
import os
from tqdm import tqdm
import time

for i in tqdm(range(1085)):
    try:
        urllib.request.urlretrieve(
            "https://picsum.photos/299/299?image="+str(i), os.path.join('background', str(i)+'.jpg'))
    except urllib.error.HTTPError as err:
        continue
