import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_dataset():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # CK+ dataset URL
    url = 'http://www.jeffcohn.net/Resources/'
    filename = 'data/ckplus.zip'
    
    try:
        print("Downloading CK+ dataset...")
        download_file(url, filename)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data')
        
        # Clean up zip file
        os.remove(filename)
        
        print("Dataset downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    download_dataset() 