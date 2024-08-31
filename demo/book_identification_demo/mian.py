import os
import cv2
import time
import json
import boto3
import torch
import shutil
import random

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from clip import CLIP

# Setting up devices
sam_device = 'cuda:0'
clip_device = 'cuda:0'

# Setting Up Paths
label_embeds_path = "GoodReads"
number_of_embedding_files = 24

clip_load_path = 'models/clip_text_on_synthetic.pth'


ocr_tmp_path = 'tmp/'
os.makedirs(ocr_tmp_path, exist_ok=True)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(results_dir + '/ocr', exist_ok=True)
os.makedirs(results_dir + '/preds', exist_ok=True)
os.makedirs(results_dir + '/flags', exist_ok=True)
os.makedirs(results_dir + '/masks', exist_ok=True)
os.makedirs(results_dir + '/images', exist_ok=True)
os.makedirs(results_dir + '/raw_books', exist_ok=True)

# Setting Up Amazon Rekognition OCR
key="YOUR KEY HERE"
secret_key="YOUR SECRET KEY HERE'

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        
    return data
        
def save_json(file_path, data):
    with open(file_path, 'w+') as json_file:
        json.dump(data, json_file)
        
def parse_response(rekognition_response, verbose=False):

    textDetections=rekognition_response['TextDetections']        
    return textDetections

def get_rekognition_scene_text(image_list, verbose=True):

    rekognition = boto3.client('rekognition', aws_access_key_id=key, aws_secret_access_key=secret_key, region_name='eu-west-1') 
    error_images = []

    image_list = tqdm(image_list) if verbose else image_list
    for img_name in image_list:

        scene_text_file_path = ocr_tmp_path + img_name.split('/')[-1].replace('.jpg', '.json')
        scene_text_subdir_path = ocr_tmp_path
        
        if not os.path.exists(scene_text_file_path):
            
            with open(img_name, 'rb') as image:
                imageBytes = bytearray(image.read())

            response = rekognition.detect_text(Image={'Bytes': imageBytes})
            scene_texts = parse_response(response, verbose=False)


            if not os.path.isdir(scene_text_subdir_path):
                os.mkdir(scene_text_subdir_path)

            save_json(scene_text_file_path, scene_texts)
            
    return error_images

def get_ocr(image_path):
    ocr = []
    num_new_words = 100
    tmp_image_path = image_path
    while num_new_words == 100:
        img = Image.open(tmp_image_path)
        draw = ImageDraw.Draw(img)
        for word in ocr:
            draw.rectangle([(word['Geometry']['BoundingBox']['Left']*img.width,
                            word['Geometry']['BoundingBox']['Top']*img.height),
                            (word['Geometry']['BoundingBox']['Left']*img.width + word['Geometry']['BoundingBox']['Width']*img.width,
                            word['Geometry']['BoundingBox']['Top']*img.height + word['Geometry']['BoundingBox']['Height']*img.height)],
                            fill='black', width=2)
        tmp_image_path = f'tmp/tmp_{random.randint(0,10000)}.jpg'
        img.save(tmp_image_path, 'JPEG', quality=80)
        error_images = get_rekognition_scene_text([tmp_image_path], verbose=True)
        new_words = [word for word in load_json(tmp_image_path.replace('.jpg', '.json')) if word['Type'] == 'WORD']
        ocr += new_words
        num_new_words = len(new_words)
    
    np.save(f'{results_dir}/ocr/{image_path.split("/")[-1].replace(".jpg", "")}.npy', ocr)
    return ocr


# Initialize Segment Anything Model
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["default"](checkpoint="models/sam_vit_h_4b8939.pth")
sam.to(device=sam_device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=10000,
)

def generate_masks(shelve_image, ocr):
    image = np.array(shelve_image.convert('RGB')) 
    image = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))

    raw_masks = mask_generator.generate(image)

    all_masks = [mask['segmentation'] for mask in raw_masks]

    # in pixels
    threshold = 1000
    to_small_threshold = 1000

    masks = [all_masks.pop(0)]
    while len(all_masks) > 0:
        mask = all_masks.pop(0)
        if np.sum(mask) < to_small_threshold:
            continue
        cont = True
        for i, mask2 in enumerate(masks):
            if np.sum(mask * mask2) > threshold:
                mask = mask + mask2
                masks[i] = mask
                cont = False
                break
        
        if cont:
            for i in range(len(all_masks)):
                if np.sum(mask * all_masks[i]) > threshold:
                    mask = mask + all_masks[i]
                    all_masks.pop(i)
                    break
            masks.append(mask)
    
    int_mask = masks[0].astype(np.uint8)
    for i, mask in enumerate(masks[1:]):
        int_mask[int_mask==0] += mask[int_mask==0].astype(np.uint8)*(i+2)

    # for each word find the mask it belongs to
    books = {i: [] for i in np.unique(int_mask)}

    for word in ocr:
        box = [int(word['Geometry']['BoundingBox']['Left']*image.shape[1]),
            int(word['Geometry']['BoundingBox']['Top']*image.shape[0]),
            int(word['Geometry']['BoundingBox']['Width']*image.shape[1]),
            int(word['Geometry']['BoundingBox']['Height']*image.shape[0])]
        unique, counts = np.unique(int_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], return_counts=True)
        mask = unique[counts.argmax()]
        books[mask] += [{'text': word['DetectedText'], 'coordinates': [word['Geometry']['BoundingBox']['Left'],
                                                                    word['Geometry']['BoundingBox']['Top'],
                                                                    word['Geometry']['BoundingBox']['Width'],
                                                                    word['Geometry']['BoundingBox']['Height']]}]
    
    return books, int_mask


# Initialize CLIP
model_config = {'CLIP_type': 'Text', 'from_pretrained': 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K', 'load_path': clip_load_path}
device = clip_device
model = CLIP(model_config)
model.eval()
model.to(device)

single_file = False
labels = pd.read_json('data/goodreads_books_with_authors.json', lines=True)

def get_preds(batch):
    with torch.no_grad():
        batch = model.preprocess(batch, device)
        ocr_embeddings = model.text_embeddings(batch).cpu().squeeze(0)
    
    ocr_embeddings = ocr_embeddings / ocr_embeddings.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.model.logit_scale.exp().cpu()
    
    if single_file:
        logits_per_ocr = torch.matmul(ocr_embeddings, labels_embeddings.t()) * logit_scale
        
        return [{'mask_id': mask_id, 'pred': pred} for mask_id, pred in zip(books.keys(), labels[logits_per_ocr.argmax(dim=1)])]

    else:
        logits_argmax = torch.ones((len(batch['labels'])), dtype=torch.long) * -1
        logits_max = torch.zeros((len(batch['labels'])))
        seen_samples = 0
        for i in range(number_of_embedding_files):
            labels_embeddings = torch.load(f'data/embeddings/GoodReads_books_embeddings_text_clip_{i}.pt')
            
            logits_per_ocr = torch.matmul(ocr_embeddings, labels_embeddings.t()) * logit_scale
            
            m = logits_per_ocr.max(dim=1).values > logits_max
            logits_max[m] = logits_per_ocr.max(dim=1).values[m]
            logits_argmax[m] = logits_per_ocr.argmax(dim=1)[m] + seen_samples
            
            seen_samples += labels_embeddings.shape[0]
        
        return [{'mask_id': mask_id, 'pred': pred} for mask_id, pred in zip(books.keys(), labels.iloc[logits_argmax].values)]


import argparse

parser = argparse.ArgumentParser(description='Book Identification Demo')
parser.add_argument('--image_path', type=str, help='Path to the image')
args = parser.parse_args()
image_path = args.image_path
image_name = image_path.split('/')[-1]

print(f"Getting OCR for {image_name}")
ocr = get_ocr(image_path)
print(f"Generating masks for {image_name}")
books, mask = generate_masks(Image.open(image_path), ocr)
np.save(f'{results_dir}/masks/{image_name.replace(".jpg", "")}.npy', mask)
np.save(f'{results_dir}/raw_books/{image_name.replace(".jpg", "")}.npy', books)
books_ocr = {'labels': [" ".join([word['text'] for word in book]) for book in books.values()]}
print(f"Getting book predictions for {image_name}")
preds = get_preds(books_ocr)
np.save(f'{results_dir}/preds/{image_name.replace(".jpg", "")}.npy', preds)          
shutil.copyfile(f"{image_path}", f"{results_dir}/images/{image_name}")
print(f"Done with {image_name}")
print('==='*10)


preds = np.load(f"{results_dir}/preds/{image_name.replace('.jpg', '')}.npy", allow_pickle=True)

image = cv2.imread(f"{results_dir}/images/" + image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (image.shape[1] // 5, image.shape[0] // 5))
mask = np.load(f"{results_dir}/masks/{image_name.replace('.jpg', '')}.npy", allow_pickle=True)
mask = np.array(Image.fromarray(mask).resize((image.shape[1], image.shape[0])))
mask_ids = np.unique(mask)

np.random.seed(1)
color = np.array([30/255, 144/255, 255/255, 0.5])
norm = plt.Normalize(1,max(mask_ids))
cmap = plt.cm.tab20

fig,ax = plt.subplots(figsize=(15,8))

im = plt.imshow(image)

plt.axis('off')
plt.tight_layout()

fig.canvas.header_visible = False

def update_annot(mask_id, pos):
    if pos[0]/image.shape[1] > 0.90:
        annot = ax.annotate("", xy=(0,0), xytext=(-180, 20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    else:
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.xy = pos
    book = preds[mask_id]
    text = f"Title: {book['pred'][1]}\nAuthor: {', '.join(book['pred'][2])}"
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(mask_id)))
    annot.get_bbox_patch().set_alpha(0.8)

    plt.imshow(image)
    inverse_mask = mask != mask_id
    h, w = inverse_mask.shape[-2:]
    mask_image = inverse_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    d_inverse_mask = ax.imshow(mask_image)

def on_click(event):
    if event.inaxes == ax:
        mask_id = mask[int(event.ydata), int(event.xdata)]
        if mask_id in mask_ids:
            plt.cla()
            plt.axis('off')
            update_annot(mask_id, (event.xdata, event.ydata))
            fig.canvas.draw_idle()
        else:
            plt.cla()
            plt.axis('off')
            plt.imshow(image)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
