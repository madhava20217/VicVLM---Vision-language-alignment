import json
import os
from PIL import Image
import re
from torch.utils.data import Dataset
import random
import math
import numpy as np
import torch
import cv2
# taken from albef

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      

class TextMaskGenerator:
    def __init__(self, masking_ratio = 0.25, mask_token = '[MASK]'):
        self.masking_ratio = masking_ratio
        self.mask_token = mask_token
        
    def __call__(self, text):
        text = np.array(text.split())  # tokenized
        len_txt = len(text)
        
        n_to_mask = math.ceil(len_txt * self.masking_ratio)
        rankings = np.random.randn(len_txt)
        
        indices = np.argpartition(rankings, -n_to_mask)[-n_to_mask:]
        text[indices] = self.mask_token
        return " ".join(text)
    
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask



class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30,
                 tokenizer = None,
                 input_size = 224,
                 mask_patch_size = 32,
                 model_patch_size = 16,
                 masking_ratio = 0.75,
                 txt_masking_ratio = 0.25,
                 mask_token = '[MASK]',
                 mask_token_id = 103,
                 max_length = 35):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.txt_masker = TextMaskGenerator(txt_masking_ratio, mask_token)
        self.img_masker = MaskGenerator(input_size,mask_patch_size,model_patch_size,masking_ratio)
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
            
        mask_caption = self.txt_masker(caption)
        
        tok_masked_txt = self.tokenizer(mask_caption, truncation = True, padding = 'max_length', max_length = self.max_length, return_token_type_ids=False)
        txt = self.tokenizer(caption, truncation = True,padding = 'max_length', max_length = self.max_length, return_token_type_ids = False)
        
        toks, attn_mask = txt['input_ids'], txt['attention_mask']
        masked_toks, masked_attn_mask = tok_masked_txt['input_ids'], tok_masked_txt['attention_mask']
        
        toks, attn_mask, masked_toks, masked_attn_mask = torch.tensor(toks), torch.tensor(attn_mask), torch.tensor(masked_toks), torch.tensor(masked_attn_mask)
        
        # masked indices
        mask_indices = (masked_toks == self.mask_token_id)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        
        img_mask = self.img_masker()
                
        return image, img_mask, toks, attn_mask, masked_toks, masked_attn_mask, mask_indices
            
            
class pretrain_dataset_regvlm(Dataset):
    # TODO extension: wSimMIM
    def __init__(self, ann_file, transform, max_words=30,
                 tokenizer = None,
                 input_size = 224,
                 mask_patch_size = 32,
                 model_patch_size = 16,
                 masking_ratio = 0.75,
                 txt_masking_ratio = 0.25,
                 mask_token = '[MASK]',
                 mask_token_id = 103,
                 max_length = 35):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.txt_masker = TextMaskGenerator(txt_masking_ratio, mask_token)
        self.img_masker = MaskGenerator(input_size,mask_patch_size,model_patch_size,masking_ratio)
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
            
        mask_caption = self.txt_masker(caption)
        
        tok_masked_txt = self.tokenizer(mask_caption, truncation = True, padding = 'max_length', max_length = self.max_length, return_token_type_ids=False)
        txt = self.tokenizer(caption, truncation = True,padding = 'max_length', max_length = self.max_length, return_token_type_ids = False)
        
        toks, attn_mask = txt['input_ids'], txt['attention_mask']
        masked_toks, masked_attn_mask = tok_masked_txt['input_ids'], tok_masked_txt['attention_mask']
        
        toks, attn_mask, masked_toks, masked_attn_mask = torch.tensor(toks), torch.tensor(attn_mask), torch.tensor(masked_toks), torch.tensor(masked_attn_mask)
        
        # masked indices
        mask_indices = (masked_toks == self.mask_token_id)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        
        img_mask = self.img_masker()
                
        return image, img_mask, toks, attn_mask, masked_toks, masked_attn_mask, mask_indices