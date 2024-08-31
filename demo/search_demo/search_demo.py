import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
matplotlib.use('agg')

images_path = "../../data/images/"
book_preds = np.load("book_preds.npy", allow_pickle=True)

def cut_spine_image(image, mask):
    """Cuts the image using the mask"""
    #Find the corners of the mask, aka the leftmost, rightmost, topmost and bottommost points
    #Then crop the image using these points
    leftmost = np.min(np.where(mask)[1])
    rightmost = np.max(np.where(mask)[1])
    topmost = np.min(np.where(mask)[0])
    bottommost = np.max(np.where(mask)[0])
    #return image[topmost:bottommost, leftmost:rightmost]
    
    mask = Image.fromarray(~mask).resize((image.size[0], image.size[1]), resample=Image.Resampling.NEAREST)
    image.paste(im=mask, box=(0, 0), mask=mask)

    return image.crop((leftmost*10, topmost*10, rightmost*10, bottommost*10))

def get_masked_image(sample):
    image = cv2.imread(images_path + sample["image_name"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))

    mask = sample["mask"] != sample["mask_id"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image)
    color = np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.axis('off')

    return fig

def get_spine_image(sample):
    image = Image.open(images_path + sample["image_name"])
    spine_image = cut_spine_image(image, sample['mask']==sample['mask_id'])
    return spine_image.rotate(-90, expand=True)


def format_book_title(title):
    return f'{title.split(" | ")[0]}: {title.split(" | ")[1]} - {title.split(" | ")[2]}'

def find_book(title):
     for book in book_preds:
        if title.lower() in ": ".join(book['pred'].lower().split(" | ")[:-1]):
            masked_image = get_masked_image(book)
            spine_image = get_spine_image(book)
            title = format_book_title(book['pred'])
            return book['image_name'], masked_image, spine_image, title
             

with gr.Blocks() as app:
    gr.Markdown("# Located Books in the Library")
    with gr.Row():
        title_input = gr.Textbox(label="Book Title")
        title_search = gr.Button("Search")
    with gr.Row():
        shelve_img = gr.Plot(label="Shelve")
    with gr.Row():
        book_img = gr.Image(type='pil', label="Book")
    with gr.Row():
        book_title = gr.Textbox(label="Book Title")
        img_id = gr.Textbox(label="Image ID")
    
    title_search.click(find_book, inputs=[title_input], outputs=[img_id, shelve_img, book_img, book_title])
    title_input.submit(find_book, inputs=[title_input], outputs=[img_id, shelve_img, book_img, book_title])


app.launch()