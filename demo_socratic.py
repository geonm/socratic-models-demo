# for demo
import os
from flask import Flask, request, session, json, Response, render_template, abort, send_from_directory
import requests
from urllib.request import urlopen
from io import BytesIO
import uuid
import time
import argparse
import torch
import clip
import utils
import csv

#os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU mode

# flask
app = Flask(__name__)
logger = app.logger
logger.info('init demo app')

# config
parser = argparse.ArgumentParser()

## flask demo parameter
parser.add_argument('--port', default=5000, type=int,
        help='This demo will be running on http://0.0.0.0:port/')
parser.add_argument('--openai-API-key', default=None, type=str,
        help='You can get an openai API key for free. See https://beta.openai.com/account/api-keys')


class Model:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('\n\tLoading VML (CLIP ViT-L/14)...')
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()
        print('\n\tLoading classifier.')
        self.openimage_classifier_weights = torch.load('./prompts/clip_ViTL14_openimage_classifier_weights.pt', map_location=self.device).type(torch.FloatTensor)
        self.openimage_classnames = self.load_openimage_classnames('./prompts/openimage-classnames.csv')
        self.tencentml_classifier_weights = torch.load('./prompts/clip_ViTL14_tencentml_classifier_weights.pt', map_location=self.device).type(torch.FloatTensor)
        self.tencentml_classnames = self.load_tencentml_classnames('./prompts/tencent-ml-classnames.txt')
        self.place365_classifier_weights = torch.load('./prompts/clip_ViTL14_place365_classifier_weights.pt', map_location=self.device).type(torch.FloatTensor)
        self.place365_classnames = self.load_tencentml_classnames('./prompts/place365-classnames.txt')

        img_types = ['photo', 'cartoon', 'sketch', 'painting']
        ppl_texts = ['no people', 'people']
        ifppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']

        self.imgtype_classifier_weights, self.imgtype_classnames = self.build_simple_classifier(img_types, lambda c: f'This is a {c}.')
        self.ppl_classifier_weights, self.ppl_classnames = self.build_simple_classifier(ppl_texts, lambda c: f'There are {c} in this photo.')
        self.ifppl_classifier_weights, self.ifppl_classnames = self.build_simple_classifier(ifppl_texts, lambda c: f'There {c} in this photo.')

    def build_simple_classifier(self, text_list, template):
        with torch.no_grad():
            texts = [template(text) for text in text_list]
            text_inputs = clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features, {idx: text for idx, text in enumerate(text_list)}

    def load_openimage_classnames(self, csv_path):
        csv_data = open(csv_path)
        csv_reader = csv.reader(csv_data)
        classnames = {idx: row[-1] for idx, row in enumerate(csv_reader)}
        return classnames

    def load_tencentml_classnames(self, txt_path):
        txt_data = open(txt_path)
        lines = txt_data.readlines()
        classnames = {idx: line.strip() for idx, line in enumerate(lines)}
        return classnames

    def zeroshot_classifier(self, image):
        '''
        image: bin image
        '''
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        #image_features = image_features.to(self.openimage_classifier_weights.dtype)
        sim = (100.0 * image_features @ self.openimage_classifier_weights.T).softmax(dim=-1)
        openimage_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(10)]
        openimage_classes = [self.openimage_classnames[idx] for idx in indices]

        sim = (100.0 * image_features @ self.tencentml_classifier_weights.T).softmax(dim=-1)
        tencentml_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(10)]
        tencentml_classes = [self.tencentml_classnames[idx] for idx in indices]

        sim = (100.0 * image_features @ self.place365_classifier_weights.T).softmax(dim=-1)
        place365_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(10)]
        place365_classes = [self.place365_classnames[idx] for idx in indices]

        sim = (100.0 * image_features @ self.imgtype_classifier_weights.T).softmax(dim=-1)
        imgtype_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(len(self.imgtype_classnames))]
        imgtype_classes = [self.imgtype_classnames[idx] for idx in indices]

        sim = (100.0 * image_features @ self.ppl_classifier_weights.T).softmax(dim=-1)
        ppl_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(len(self.ppl_classnames))]
        ppl_classes = [self.ppl_classnames[idx] for idx in indices]

        sim = (100.0 * image_features @ self.ifppl_classifier_weights.T).softmax(dim=-1)
        ifppl_scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(len(self.ifppl_classnames))]
        ifppl_classes = [self.ifppl_classnames[idx] for idx in indices]

        return image_features, openimage_scores, openimage_classes, tencentml_scores, tencentml_classes,\
               place365_scores, place365_classes, imgtype_scores, imgtype_classes,\
               ppl_scores, ppl_classes, ifppl_scores, ifppl_classes

    def sorting_texts(self, image_features, captions):
        with torch.no_grad():
            text_inputs = clip.tokenize(captions).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sim = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores, indices = [drop_gpu(tensor) for tensor in sim[0].topk(len(captions))]
            sorted_captions = [captions[idx] for idx in indices]

        return scores, sorted_captions

def drop_gpu(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def init_worker(args):
    global model
    model = Model(args)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


@app.route('/', methods=['POST'])
def index_post():
    request_start = time.time()
    configs = request.form
            
    session_id = str(uuid.uuid1())

    image_url = configs['image_url']
    image = utils.read_image_from_url(image_url)
    
    image_features, openimage_scores, openimage_classes, tencentml_scores, tencentml_classes, place365_scores, place365_classes, imgtype_scores, imgtype_classes, ppl_scores, ppl_classes, ifppl_scores, ifppl_classes = model.zeroshot_classifier(image)

    prompt_caption, prompt_search = utils.generate_prompt(openimage_classes, tencentml_classes, place365_classes, imgtype_classes, ppl_classes, ifppl_classes)
    generated_captions = utils.generate_captions(prompt_caption, model.args.openai_API_key, num_captions=3)
    generated_keywords = utils.generate_captions(prompt_search, model.args.openai_API_key, num_captions=1)

    caption_scores, sorted_captions = model.sorting_texts(image_features, generated_captions)
    keyword_scores, sorted_keywords = model.sorting_texts(image_features, generated_keywords)

    utils.save_results(image_url,
                       openimage_scores,
                       openimage_classes,
                       tencentml_scores,
                       tencentml_classes,
                       place365_scores,
                       place365_classes,
                       imgtype_scores,
                       imgtype_classes, 
                       ppl_scores,
                       ppl_classes,
                       ifppl_scores,
                       ifppl_classes,
                       prompt_caption,
                       caption_scores,
                       sorted_captions,
                       keyword_scores,
                       sorted_keywords,
                       session_id)
    return render_template('index.html', session_id=session_id)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    args = parser.parse_args()

    init_worker(args)

    app.run(host='0.0.0.0', port=args.port)
