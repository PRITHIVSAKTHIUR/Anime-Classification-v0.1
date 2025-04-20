![SD.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/JALO6mW1WZSHjUxsotxW2.png)

# **Anime-Classification-v1.0**

> **Anime-Classification-v1.0** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify anime-related images using the **SiglipForImageClassification** architecture.

```py
Classification Report:
              precision    recall  f1-score   support

          3D     0.7979    0.8443    0.8204      4649
     Bangumi     0.8677    0.8728    0.8702      4914
       Comic     0.9716    0.9233    0.9468      5746
Illustration     0.8204    0.8186    0.8195      6064

    accuracy                         0.8648     21373
   macro avg     0.8644    0.8647    0.8642     21373
weighted avg     0.8670    0.8648    0.8656     21373
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/5jH61328ZIygBR0ExlWVv.png)

---

The model categorizes images into 4 anime-related classes:

```
    Class 0: "3D"
    Class 1: "Bangumi"
    Class 2: "Comic"
    Class 3: "Illustration"
```

---

## **Install dependencies**

```python
!pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Anime-Classification-v1.0"  # New model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_anime_image(image):
    """Predicts the anime category for an input image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "3D", "1": "Bangumi", "2": "Comic", "3": "Illustration"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_anime_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Anime Classification v1.0",
    description="Upload an image to classify the anime style category."
)

if __name__ == "__main__":
    iface.launch()
```

---

## **Intended Use:**

The **Anime-Classification-v1.0** model is designed to classify anime-related images. Potential use cases include:

- **Content Tagging:** Automatically label anime artwork on platforms or apps.
- **Recommendation Engines:** Enhance personalized anime content suggestions.
- **Digital Art Curation:** Organize galleries by anime style for artists and fans.
- **Dataset Filtering:** Categorize and filter images during dataset creation.
