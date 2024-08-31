import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput

from transformers.models.clip.modeling_clip import clip_loss


class CLIP(nn.Module):
    def __init__(self, config=None):
        super(CLIP, self).__init__()
        
        if 'from_pretrained' in config.keys():
            model_repo = config['from_pretrained']
        else:
            model_repo = "openai/clip-vit-base-patch32"
            
        self.model = CLIPModel.from_pretrained(model_repo)
        self.processor = CLIPProcessor.from_pretrained(model_repo)

        if 'CLIP_type' in config.keys():
            self.CLIP_type = config['CLIP_type'].lower()
        else:
            self.CLIP_type = 'image'


        if 'load_path' in config.keys():
            if config['load_path'] is not None:
                self.load_weights(config['load_path'])

        # Initialize the Multi-headed Attention Fusion Mechanism
        if self.CLIP_type in ['image+text', 'text+image', 'image_plus_text2image_plus_text']:
            assert 'num_attention_heads' in config.keys(), "Please specify the number of attention heads for the Multi-headed Attention Fusion Mechanism"
            assert 'hidden_size' in config.keys(), "Please specify the hidden size for the Multi-headed Attention Fusion Mechanism"

            self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
            self.all_head_size = config['num_attention_heads'] * self.attention_head_size

            self.query = nn.Linear(config['hidden_size'], self.all_head_size)
            self.key = nn.Linear(config['hidden_size'], self.all_head_size)
            self.value = nn.Linear(config['hidden_size'], self.all_head_size)

            self.multihead_attn = nn.MultiheadAttention(724, config['num_attention_heads'])
        

    def load_weights(self, path):
        print(f"Loading weights from {path}")
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, batch, return_loss=True):
        # Get the embeddings of the labels
        if self.CLIP_type in ['image', 'text', 'image+text', 'text+image', 'image+text2image+text']:
            if 'labels' in batch.keys():
                labels_embeddings = self.text_embeddings(dict(labels=batch['labels']))
                labels_embeddings = labels_embeddings / labels_embeddings.norm(p=2, dim=-1, keepdim=True)
        
            elif 'labels_embeds' in batch.keys():
                labels_embeddings = batch['labels_embeds']
        
        if self.CLIP_type in ['text2image', 'image2image', 'image+text2image+text']:
            if 'label_images' in batch.keys():
                label_images_embeddings = self.image_embeddings(dict(images=batch['label_images']))
                label_images_embeddings = label_images_embeddings / label_images_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            elif 'label_images_embeds' in batch.keys():
                label_images_embeddings = batch['label_images_embeds']

        # Get the logits
        if self.CLIP_type == 'image':
            logits_per_label, logits_per_input = self.image2text(batch, labels_embeddings)

        elif self.CLIP_type == 'text':
            logits_per_label, logits_per_input = self.text2text(batch, labels_embeddings)
        
        elif self.CLIP_type == 'image+text' or self.CLIP_type == 'text+image':
            logits_per_label, logits_per_input = self.image_plus_text2text(batch, labels_embeddings)
        
        elif self.CLIP_type == 'text2image':
            logits_per_label, logits_per_input = self.text2image(batch, label_images_embeddings)
        
        elif self.CLIP_type == 'image2image':
            logits_per_label, logits_per_input = self.image2image(batch, label_images_embeddings)
        
        elif self.CLIP_type == 'image+text2image+text':
            logits_per_label, logits_per_input = self.image_plus_text2image_plus_text(batch, labels_embeddings, label_images_embeddings)

        # Calculate the loss
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_label)

        return CLIPOutput(
            loss=loss,
            logits_per_text=logits_per_label,
            logits_per_image=logits_per_input
        )

    
    def text_embeddings(self, batch):
        return self.model.get_text_features(input_ids=batch['labels'])
    
    def image_embeddings(self, batch):
        return self.model.get_image_features(pixel_values=batch['images'])

    def image2text(self, batch, labels_embeddings):
        image_embeddings = self.image_embeddings(dict(images=batch['images']))
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeddings, labels_embeddings.t()) * logit_scale
        logits_per_label = logits_per_image.t()

        del image_embeddings
        return logits_per_label, logits_per_image
    
    def text2image(self, batch, label_images_embeddings):
        ocr_embeddings = self.text_embeddings(dict(labels=batch['ocr']))
        ocr_embeddings = ocr_embeddings / ocr_embeddings.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_ocr = torch.matmul(ocr_embeddings, label_images_embeddings.t()) * logit_scale
        logits_per_label = logits_per_ocr.t()

        del ocr_embeddings
        return logits_per_label, logits_per_ocr

    def text2text(self, batch, labels_embeddings):
        ocr_embeddings = self.text_embeddings(dict(labels=batch['ocr']))
        ocr_embeddings = ocr_embeddings / ocr_embeddings.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_ocr = torch.matmul(ocr_embeddings, labels_embeddings.t()) * logit_scale
        logits_per_label = logits_per_ocr.t()

        del ocr_embeddings
        return logits_per_label, logits_per_ocr

    def image2image(self, batch, label_images_embeddings):
        image_embeddings = self.image_embeddings(dict(images=batch['images']))
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeddings, label_images_embeddings.t()) * logit_scale
        logits_per_label = logits_per_image.t()

        del image_embeddings
        return logits_per_label, logits_per_image

    def image_plus_text2text(self, batch, labels_embeddings):
        ocr_embeddings = self.text_embeddings(dict(labels=batch['ocr']))
        ocr_embeddings = ocr_embeddings / ocr_embeddings.norm(p=2, dim=-1, keepdim=True)

        image_embeddings = self.image_embeddings(dict(images=batch['images']))
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Fusion of the embeddings
        fused_embeddings = self.fusion(ocr_embeddings, image_embeddings) #(ocr_embeddings + image_embeddings) / 2

        logit_scale = self.model.logit_scale.exp()
        logits_per_fused = torch.matmul(fused_embeddings, labels_embeddings.t()) * logit_scale
        logits_per_label = logits_per_fused.t()
        
        del ocr_embeddings, image_embeddings, fused_embeddings
        return logits_per_label, logits_per_fused

    def image_plus_text2image_plus_text(self, batch, label_embeddings, label_images_embeddings):
        ocr_embeddings = self.text_embeddings(dict(labels=batch['ocr']))
        ocr_embeddings = ocr_embeddings / ocr_embeddings.norm(p=2, dim=-1, keepdim=True)

        image_embeddings = self.image_embeddings(dict(images=batch['images']))
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        fused_query_embeddings = self.fusion(ocr_embeddings, image_embeddings) #(ocr_embeddings + image_embeddings) / 2

        fused_label_embeddings = self.fusion(label_embeddings, label_images_embeddings) #(label_embeddings + label_images_embeddings) / 2

        logit_scale = self.model.logit_scale.exp()
        logits_per_fused = torch.matmul(fused_query_embeddings, fused_label_embeddings.t()) * logit_scale
        logits_per_label = logits_per_fused.t()

        del ocr_embeddings, image_embeddings, fused_query_embeddings, fused_label_embeddings
        return logits_per_label, logits_per_fused
    
    def fusion(self, embeddings1, embeddings2):
        return self.multihead_attn(self.query(embeddings1), self.key(embeddings2), self.value(embeddings2))[0]
    
    def preprocess(self, batch, device):
        batch = {k: v for k, v in batch.items()}
        
        if 'labels' in batch.keys():
            batch['labels'] = self.processor(batch['labels'], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        elif 'labels_embeds' in batch.keys():
            batch['labels_embeds'] = batch['labels_embeds'].to(device)
        
        if 'label_images' in batch.keys():
            batch['label_images'] = batch['label_images'].to(device)
        elif 'label_images_embeds' in batch.keys():
            batch['label_images_embeds'] = batch['label_images_embeds'].to(device)
        
        if 'ocr' in batch.keys():
            batch['ocr'] = self.processor(text=batch['ocr'], return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
 
        if 'images' in batch.keys():
            batch['images'] = batch['images'].to(device)

        
        return batch