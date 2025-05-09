import torch
import torch.nn as nn
import torchvision.models as models
import os
import gdown

def download_pretrained_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Google Drive file ID for the pre-trained model
    file_id = '1-0B3HWUXuUum1RENiQjNlN0Ztdm9ROHc'
    output = 'models/emotion_model.pth'
    
    try:
        # Download the model
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        print(f"Model downloaded successfully to {output}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_pretrained_model() 