## Simple interactive web-based demo for Socratic models
+ [paper](https://arxiv.org/abs/2204.00598) [official repository](https://github.com/google-research/google-research/tree/master/socraticmodels) [official project website](https://socraticmodels.github.io/)
+ This is an un-official repository for simple interactive web-based demo for socratic models.
+ This demo produces captions and keywords for image search, highly related to an input image.
+ This repo contains precomputed zero-shot classifiers using CLIP ViT-L/14 model.
  + classifier for object classifier using class names from [tencent-ML-images](https://github.com/Tencent/tencent-ml-images/blob/master/data/dictionary_and_semantic_hierarchy.txt)
  + classifier for place classifier from [Place365](http://places2.csail.mit.edu/)
  + classifier for additional object classifier using class names from [openimage](https://storage.googleapis.com/openimages/web/download.html)

### prompt for image captioning
```python
    prompt_caption = f'''I am an intelligent image captioning bot.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    A creative short caption I can generate to describe this image is:'''
```

### prompt for keyword generation
```python
    prompt_search = f'''Let's list keywords that include the following description.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    Relevant keywords which we can list and are seperated with comma are:'''
```

## Overview
### Get your OpenAI API Key for GPT3
+ Don't worry about it. It's free.
+ See [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)

### How to run the demo
```
$ python demo_socratic.py --port 5000 --openai-API-key {YOUR_OpenAI_API_KEY}
```
+ Demo will be [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

### How to use the demo
+ Just fetch an image url.
howtouse1

### Result
+ url: https://image.shutterstock.com/image-photo/man-climbing-mountain-260nw-613489679.jpg
+ generated caption:

IMAGE IMAGE IMAGE


