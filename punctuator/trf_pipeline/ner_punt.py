import json
import os.path
from transformers import pipeline, TokenClassificationPipeline
from seqeval import metrics
import pandas as pd
from tqdm import tqdm

from punctuator.utils import preprocess_text, tokenize_words, text2labels, transform_sentences

BASE_DATASET = "../../datasets/"
BASE_MODEL_DIR = "../../punctuator/trf_pipeline/models/ner_punt"


def get_classifier():
    return pipeline("ner", model=BASE_MODEL_DIR, aggregation_strategy="average")


def predict(text_, split_mode='sentence', max_len=512, overlap=20):
    if split_mode == 'sentence':
        overlap = 0
    texts = preprocess_text(text_, split_mode, max_len, overlap)
    classifier = get_classifier()
    outputs = []
    new_text = ''
    for i, text in enumerate(texts):
        outs = classifier(text)
        outputs.extend(outs)
        if len(texts) >= 2 and i < (len(texts) - 1) and overlap > 0:
            tokens = tokenize_words(text)[:-overlap]
            text = ' '.join(tokens)
        new_text += ' ' + transform_sentences(text, outs)

    return text2labels(new_text)


def main():
    annotator1 = json.load(open(os.path.join(BASE_DATASET, "annotator1.json"), "r"))

    bert_labels = []
    for item in tqdm(annotator1, total=len(annotator1)):

        text_id = item["text_id"]

        ann_text = item["text"]
        bert_label = predict(ann_text, split_mode='max_len', max_len=512)
        bert_labels.append(bert_label)

        if len(bert_label) != len(item['labels']):
            print()
            print(item['text_id'])
            print(item["text"])
            print(len(bert_label), len(item['labels']))
            print(tokenize_words(item["text"]))
            break
