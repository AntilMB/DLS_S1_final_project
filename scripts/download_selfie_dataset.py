import subprocess
from glob import glob
import shutil
import pandas as pd
import requests
import os
from tqdm.auto import tqdm
import random


def download_cage_dataset():
#     subprocess.run('mkdir ../datasets/selfie_dataset/trainB/', shell=True)

    files = pd.read_csv('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt', 
                        skiprows=2, 
                        sep='\t',
                        names=['person', 'imagenum', 'url', 'rect', 'md5sum']
                       )
    files = files[files['person'] == 'Nicolas Cage']
    
    for url in tqdm(files['url']):
        name = url.split('/')[-1]
        path = f'../datasets/selfie_dataset/trainB/{name}'
        
        if os.path.exists(path):
            continue
        
        try:
            img_data = requests.get(url, timeout=10).content
            with open(path, 'wb') as handler:
                handler.write(img_data)
        except:
            print(name, url)
            

def selfie_dataset():
    subprocess.run('wget --no-check-certificate -P ../datasets/ https://www.crcv.ucf.edu/data/Selfie/Selfie-dataset.tar.gz', shell=True)
    subprocess.run('mkdir ../datasets/selfie_dataset', shell=True)
    subprocess.run('tar -xvf ../datasets/Selfie-dataset.tar.gz -C ../datasets/selfie_dataset/', shell=True)
    subprocess.run('mkdir ../datasets/selfie_dataset/trainA/', shell=True)

    files = glob('../datasets/selfie_dataset/Selfie-dataset/images/*')
    print(len(files))

    for file in tqdm(files):
        new = file.replace('Selfie-dataset/images', 'trainA')
        shutil.move(file, new)

    subprocess.run('rm -rf ../datasets/selfie_dataset/Selfie-dataset', shell=True)


def make_test(path, type_):
    subprocess.run(f'mkdir {path}/test{type_}', shell=True)
    files = glob(f'{path}/train{type_}/*')
    files_to_move = random.choices(files, k=int(len(files) / 10))
    
    for old_path in tqdm(files_to_move):
        new_path = old_path.replace('train', 'test')
        try:
            shutil.move(old_path, new_path)
        except:
            print(old_path, new_path)


# selfie_dataset()
# download_cage_dataset()
make_test('../datasets/selfie_dataset/', 'A')
make_test('../datasets/selfie_dataset/', 'B')
