import gradio as gr
from fastai.vision.all import *

__all__ = ['learn','category','classify_images','image','label','examples','demo'import gradio as gr
from fastai.vision.all import *

__all__ = ['learn','categories','classify_images','image','label','examples','demo']

learn = load_learner('bear_image_detection_model.pkl')

categories = ('Blackbear','Grizzlybear','Teddybear')

def classify_images(img_path):
    im = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(im)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil", label="Input Image")
label = gr.Label(label="Prediction")

examples = [
    "data/bears/grizzlybear/Image_1.jpg",
    "data/bears/teddybear/Image_1.jpg",
    "data/bears/blackbear/Image_1.jpg",
    "data/cats/Image_1.jpg"
]

demo = gr.Interface(fn=classify_images,
                    inputs=image,
                    outputs=label,
                    examples=examples)

demo.launch()

