import gradio as gr
import numpy as np
from transformers import pipeline

title = "Token Classification"
description = """
Label the entities of a sentence as: 
1. person(PER), 
2. organization(ORG), 
3. location(LOC) 
4. miscellaneous(MISC).
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [my github repository](https://github.com/Neural-Net-Rahul/P2-Token-Classification-using-Fine-tuned-Hugging-face-transformer) and my [fine tuned model](https://huggingface.co/neural-net-rahul/bert-finetuned-ner)"

textbox = gr.Textbox(label="Type your sentence here :", placeholder="My name is Bill Gates.", lines=3)

model = pipeline('token-classification',model='neural-net-rahul/bert-finetuned-ner')

def predict(text):
  result = []
  word1 = None
  entity_past = None
  for dicti in model(text):
    entity,word = dicti['entity'],dicti['word']
    if entity[0]=='B':
      if word1 is not None:
        if entity_past =='B-PER':
          entity_past = 'Person'
        elif entity_past =='B-ORG':
          entity_past = 'Organization'
        elif entity_past =='B-MISC':
          entity_past = 'Miscellaneous'
        elif entity_past =='B-LOC':
          entity_past = 'Location'
        result.append([word1,entity_past])
      word1 = word;
      entity_past = entity;
    else:
      word1 = word1 + word.lstrip("#");
  if entity_past =='B-PER':
    entity_past = 'Person'
  elif entity_past =='B-ORG':
    entity_past = 'Organization'
  elif entity_past =='B-MISC':
    entity_past = 'Miscellaneous'
  elif entity_past =='B-LOC':
    entity_past = 'Location'
  result.append([word1,entity_past])
  return result

gr.Interface(
    fn=predict,
    inputs=textbox,
    outputs=[gr.Text()],
    title=title,
    description=description,
    article=article,
    examples=[["Mark founded Facebook, shaping global social media connectivity."], ["Delhi is the most beautiful state after Kerala"]],
).launch(share=True) 