import json
import os
import re
import string
import traceback
from itertools import chain

import click
import numpy as np
import pandas as pd
import spacy
import torch
from nltk.tokenize import wordpunct_tokenize
from seqeval.metrics import classification_report
from silence_tensorflow import silence_tensorflow
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertTokenizer

from punctuator.utils import split_in_sentences, tokenize_words, remove_punctuation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
silence_tensorflow()

nlp = spacy.blank('pt')
MODEL_PATH = "models/bert-portuguese-tedtalk2012"

bert_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


def text2labels(sentence):
    """
    Convert text to labels
    :param sentence: text to convert
    :return:  list of labels
    """
    tokens = wordpunct_tokenize(sentence.lower())

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                labels.append('O')
            elif token in ['.', '?', '!', ';']:
                labels[-1] = 'I-PERIOD'
            elif token == ',':
                labels[-1] = 'I-COMMA'

        except IndexError:
            raise ValueError(f"Sentence can't start with punctuation {token}")
    return labels


def merge_dicts(dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = dict_args[0]
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def get_model(model_path,
              model_type="bert",
              labels=None,
              max_seq_length=512):
    model_args = NERArgs()

    if labels is not None:
        model_args.labels_list = labels
    else:
        model_args.labels_list = ["O", "COMMA", "PERIOD", "QUESTION"]
    model_args.silent = True
    model_args.max_seq_length = max_seq_length
    return NERModel(
        model_type,
        model_path,
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )


def split_lines(text):
    paragraphs = text.split('\n')
    return paragraphs


def preprocess_text(text):
    """
    Preprocess text for prediction
    :param text: text to preprocess
    :return:  list of preprocessed text
    """
    sentences = split_in_sentences(text)

    return list(map(lambda x: remove_punctuation(x).lower(), sentences))


def predict(test_text: str, bert_model):
    """
    Predict punctuation for text
    :param test_text:   text to predict punctuation for
    :param bert_model:  model to use for prediction
    :return:  list of predicted labels
    """
    texts = preprocess_text(test_text)

    prediction_list, raw_outputs = bert_model.predict(texts, )
    pred_dict = merge_dicts(list(chain(*prediction_list)))
    words = tokenize_words(test_text)
    words = sorted(set(words))
    pred_words = sorted(pred_dict.keys())

    if len(pred_words) != len(words):
        print("Number of tokens doesn't match")
        print("Number of tokens in text: ", len(words))
        print("Number of tokens in prediction: ", len(pred_dict))
        breakpoint()
    return get_labels(test_text, pred_dict)


def get_labels(text, pred_dict):
    labels = []
    try:
        # Tokenização do BERT tá diferente daque é feita aqui

        tokens = wordpunct_tokenize(text.lower())

        for word in tokens:
            if word not in string.punctuation:
                if pred_dict[word] == "QUESTION":
                    label = "I-PERIOD"
                elif pred_dict[word] == "COMMA":
                    label = "I-COMMA"
                elif pred_dict[word] == "PERIOD":
                    label = "I-PERIOD"
                else:
                    label = "O"
                labels.append(label)
    except KeyError:
        print("KeyError", pred_dict)
        print(traceback.format_exc())
        print(len(bert_tokenizer.tokenize(text)))
        print(text)
        breakpoint()
    return labels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@click.command()
@click.option('--text', '-t', help='Text to predict punctuation for')
def main(text=None):
    model = get_model(MODEL_PATH)
    labels = predict(text, model)
    print(labels)


if __name__ == '__main__':

    model = get_model(MODEL_PATH, model_type="bert", max_seq_length=512)

    DATA_PATH = "../dataset/"

    annotator1 = json.load(open("../dataset/annotator1.json", "r"))
    annotator2 = json.load(open("../dataset/annotator2.json", "r"))
    bert_annots = []
    both_annotator = json.load(open("../dataset/both_anotators.json", "r"))
    dataset = {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "both_annotator": both_annotator
    }
    bert_labels = []

    for annt1, annt2, item in zip(annotator1, annotator2, both_annotator):
        text_id = annt2["text_id"]
        print("Processing Text ID: ", text_id)

        ann_text = annt2["text"].lower()
        bert_label = predict(ann_text, model)

        bert_labels.append(bert_label)

        item.pop("ents")
        item.pop("labels")
        bert_annotation = item
        bert_annotation["labels"] = bert_label
        bert_annots.append(bert_annotation)
    with open("../../bert_annotations/annotator2/bert_annotations_sentences.json", "w") as f:
        json.dump(bert_annots, f, cls=NpEncoder, indent=4)

    for data_label in dataset:
        true_labels = []
        items = dataset[data_label]

        for item in items:
            true_labels.append(item["labels"])

        report = classification_report(true_labels, bert_labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f"../dataset/{data_label}_bert_report.csv")
