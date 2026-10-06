---
license: "cc-by-nc-4.0"
tags:
- vision
- video-classification
---

# VideoMAE (base-sized model, fine-tuned on Kinetics-400) 

VideoMAE model pre-trained for 1600 epochs in a self-supervised way and fine-tuned in a supervised way on Kinetics-400. It was introduced in the paper [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) by Tong et al. and first released in [this repository](https://github.com/MCG-NJU/VideoMAE). 

Disclaimer: The team releasing VideoMAE did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

VideoMAE is an extension of [Masked Autoencoders (MAE)](https://arxiv.org/abs/2111.06377) to video. The architecture of the model is very similar to that of a standard Vision Transformer (ViT), with a decoder on top for predicting pixel values for masked patches.

Videos are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds fixed sinus/cosinus position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of videos that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled videos for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire video.

## Intended uses & limitations

You can use the raw model for video classification into one of the 400 possible Kinetics-400 labels.

### How to use

Here is how to use this model to classify a video:

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

video = list(np.random.randn(16, 3, 224, 224))

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

inputs = processor(video, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/videomae.html#).

## Training data

(to do, feel free to open a PR)

## Training procedure

### Preprocessing

(to do, feel free to open a PR)

### Pretraining

(to do, feel free to open a PR)

## Evaluation results

This model obtains a top-1 accuracy of 80.9 and a top-5 accuracy of 94.7 on the test set of Kinetics-400.

### BibTeX entry and citation info

```bibtex
misc{https://doi.org/10.48550/arxiv.2203.12602,
  doi = {10.48550/ARXIV.2203.12602},
  url = {https://arxiv.org/abs/2203.12602},
  author = {Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```