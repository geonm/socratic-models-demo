import os
import requests
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
import json
import openai

def read_image_from_url(image_url):
    response = requests.get(image_url, stream=True)
    return Image.open(BytesIO(response.content))

def postprocess_cls(scores, classes):
    scores = [float('%.4f' % float(val)) for val in scores]
    outputs = []
    for score, cls in zip(scores, classes):
        outputs.append([score, cls])
    return outputs

def save_results(image_url, openimage_scores, openimage_classes, tencentml_scores, tencentml_classes, place365_scores, place365_classes, imgtype_scores, imgtype_classes, ppl_scores, ppl_classes, ifppl_scores, ifppl_classes, generated_prompt, caption_scores, sorted_captions, keyword_scores, sorted_keywords, session_id):
    dirpath = os.path.join('static/results', session_id)
    os.makedirs(dirpath, exist_ok=True)

    rst = {}
    rst['image_url'] = image_url
    rst['openimage_results'] = postprocess_cls(openimage_scores, openimage_classes)
    rst['tencentml_results'] = postprocess_cls(tencentml_scores, tencentml_classes)
    rst['place365_results'] = postprocess_cls(place365_scores, place365_classes)
    rst['imgtype_results'] = postprocess_cls(imgtype_scores, imgtype_classes)
    rst['ppl_results'] = postprocess_cls(ppl_scores, ppl_classes)
    rst['ifppl_results'] = postprocess_cls(ifppl_scores, ifppl_classes)
    rst['caption_results'] = postprocess_cls(caption_scores, sorted_captions)
    rst['keyword_results'] = postprocess_cls(keyword_scores, sorted_keywords)
    rst['generated_prompt'] = generated_prompt
    rst['session_id'] = session_id

    with open(os.path.join(dirpath, 'results.json'), 'w') as f:
        json.dump(rst, f)

def generate_prompt(openimage_classes, tencentml_classes, place365_classes, imgtype_classes, ppl_classes, ifppl_classes):
    img_type = imgtype_classes[0]
    ppl_result = ppl_classes[0]
    if ppl_result == 'people':
        ppl_result = ifppl_classes[0]
    else:
        ppl_result = 'are %s' % ppl_result

    sorted_places = place365_classes

    object_list = ''
    for cls in tencentml_classes:
        object_list += f'{cls}, '
    for cls in openimage_classes[:2]:
        object_list += f'{cls}, '
    object_list = object_list[:-2]

    prompt_caption = f'''I am an intelligent image captioning bot.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    A creative short caption I can generate to describe this image is:'''

    prompt_search = f'''Let's list keywords that include the following description.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    Relevant keywords which we can list and are seperated with comma are:'''

    return prompt_caption, prompt_search


def generate_captions(prompt, openai_api_key, num_captions=10):
    openai.api_key = openai_api_key
    gpt_version = "text-davinci-002"
    max_tokens = 32
    temperature = 0.9
    stop=None
    gpt_results = []
    for _ in range(num_captions):
        response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        gpt_results.append(response["choices"][0]["text"].strip().replace('"', ''))
    return gpt_results


