import gradio as gr
from fastai.vision.all import *

__all__ = ['learn','categories','classify_image','image','label','examples','demo','greet'] 

learn = load_learner('bear_image_detection_model.pkl')

category = ('Blackbear','Grizzlybear','Teddybear')
def classify_images(img_path):
    im = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(im)
    return dict(zip(category,map(float,probs)))


image = gr.Image(type="pil", label="Input Image")   # For input images
label = gr.Label(label="Prediction")               # For output labels

examples = [
    "data/bears/grizzlybear/Image_1.jpg",
    "data/bears/teddybear/Image_1.jpg",
    "data/bears/blackbear/Image_1.jpg",
    "data/cats/Image_1.jpg"
]

demo = gr.Interface(
    fn=classify_images,   # your prediction function
    inputs=image,
    outputs=label,
    examples=examples
)

demo.launch()

