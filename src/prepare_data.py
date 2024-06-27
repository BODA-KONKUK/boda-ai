import torch
import skimage.io as io
from PIL import Image
import os

# modify https://github.com/dino-chiio/blip-vqa-finetune/blob/main/finetuning.py
class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, processor, img_path=""):
        self.dataset = dataset
        self.processor = processor
        self.img_path = img_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image_file = self.dataset[idx]['image']
        image_path = os.path.join(self.img_path, image_file)
        image = Image.open(image_path).convert("RGB")
        text = question
        
        encoding = self.processor(image, text, 
                                  max_length= 512, pad_to_max_length=True,
                                  # padding="max_length", truncation=True, 
                                  return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length= 8, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        
        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
            
        return encoding