import re
import string

from nltk import wordpunct_tokenize
from nltk.tokenize import regexp
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertTokenizer


import torch


def define_char_case(punct, text_list, i):
    """Define se o caracter é maiúsculo ou minúsculo"""

    if punct == '.':
        text_list[i + 2] = text_list[i + 2].upper()
        # Coloca aprimeira letra em maiúsculo
    elif punct == ',':
        text_list[i + 2] = text_list[i + 2].lower()
        # Coloca a primeira letra em minúsculo
    return text_list


def remove_extra_punctuation(ann_text_list, start_char):
    """Remove pontuação extra
    :param ann_text_list: lista de caracteres do texto
    :param start_char: índice do caracter a partir do qual a busca será feita
    :return: lista de caracteres do texto
    """
    shift = 0
    i = start_char
    char = ann_text_list[start_char]

    while char in [' ', '\n', '\t', '.', ',', ';', ':', '!', '?']:
        ann_text_list.pop(i)
        shift -= 1

        char = ann_text_list[i]

    return ann_text_list, shift


def remove_repeated_punctuation(text_list, start_char, ref_punct):
    """Remove pontuação repetida"""
    shift = 0
    for j in range(start_char, len(text_list)):
        if text_list[j] != ref_punct:
            text_list.pop(j)
            shift -= 1
        if text_list[j] in [' ', '\n', '\t']:
            break

    return text_list, shift


def fix_punctuation(ann_text_list, start_char, end_char, punct):
    other_punctuations = ['.', ',', ';', ':', '!', '?']
    other_punctuations.remove(punct)
    shift = 0
    try:
        text_span = ann_text_list[start_char:end_char][0]
    except IndexError:
        # Não há matches com o caracter do texto e então significa que o aluno esqueceu ponto final.
        if ann_text_list[-1] not in ['.', ',', ';', ':', '!', '?']:
            ann_text_list.append('.')
            shift += 1
        elif ann_text_list[-1] in other_punctuations:
            ann_text_list[-1] = '.'
        return ann_text_list, shift

    if text_span != punct:

        try:
            for i in range(start_char, end_char):
                old_char = ann_text_list[i]
                if old_char in other_punctuations:
                    ann_text_list[i] = punct
                    ann_text_list = define_char_case(punct, ann_text_list, i)
                    break
                if old_char in [' ', '\n', '\t']:
                    ann_text_list[i] = punct

                    ann_text_list.insert(i + 1, old_char)
                    shift += 1
                    ann_text_list, shift_removed = remove_extra_punctuation(ann_text_list, i + 2)
                    shift += shift_removed
                    ann_text_list = define_char_case(punct, ann_text_list, i)
                    break
        except IndexError:
            ann_text_list.append(punct)
            shift += 1

    else:

        for i in range(start_char - 1, end_char + 1):
            if ann_text_list[i] == punct:
                ann_text_list.pop(i)  # Remove pontuação extra

                # Adiciona espaço após a pontuação se necessário
                if end_char + 1 >= len(ann_text_list):
                    ann_text_list.append('.')  # Adiciona ponto final
                    break
                if ann_text_list[i] in [' ', '\n', '\t']:
                    shift -= 1
                else:
                    ann_text_list.insert(i, ' ')

                    shift += 1
                break

    return ann_text_list, shift


def replace(sentence):
    tokenizer = regexp.RegexpTokenizer(r'\w+|[.,?]')

    tokens = tokenizer.tokenize(sentence.lower())

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                # sent_data.append([sent_id,'O',token])
                labels.append('O')
            elif token in ['.', '?']:
                # sent_data[-1][1] = 'PERIOD'
                labels[-1] = 'PERIOD'
            elif token == ',':
                # sent_data[-1][1] = 'COMMA'
                labels[-1] = 'COMMA'

        except IndexError:
            continue

    return labels


def load_model(path: str, labels: list, max_length: int = 512, model_type: str = 'bert'):
    model_args = NERArgs()
    model_args.labels_list = labels

    model_args.max_seq_length = max_length
    return NERModel(
        model_type,
        path,
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )


def text2labels(sentence):
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
            print(sentence)
            print(tokens)
            raise ValueError(f"Sentence can't start with punctuation {token}")
    return labels


def remove_extra_punctuation_regex(text):
    text = re.sub(r'([.,?!;:])+', r'\1', text)
    return text


def split_in_sentences(text):
    return list(filter(lambda x: x != '', re.split(r' *[\.\?!][\'"\)\]]* *', text)))


def add_space_after_punkt(text):
    text = re.sub(r'([.,?!;:])+', r'\1 ', text)
    return text


def tokenize_words(text, remove_punctuation=True):
    """
    Tokenize words in text
    :param remove_punctuation:
    :param text: text to tokenize
    :param remove_punctuation:  remove punctuation from text
    :return:  list of tokens
    """
    if remove_punctuation:
        words = [word for word in wordpunct_tokenize(text) if word not in string.punctuation]
    else:
        words = wordpunct_tokenize(text)
    return words


def truncate_texts_by_max_len(text, max_seq_length=512, overlap=20, bert_tokenizer=None):
    """
    Truncate sentences to fit into BERT's max_seq_length
    :param bert_tokenizer:
    :param text:  text to truncate
    :param max_seq_length:  max sequence length
    :param overlap:  overlap between sentences
    :return:    list of truncated sentences
    """
    if bert_tokenizer is None:
        bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    texts = []

    tokens = tokenize_words(text)

    bert_tokens = bert_tokenizer.tokenize(text)

    len_text = ((max_seq_length * len(tokens)) // len(bert_tokens)) + 1

    if len(bert_tokens) > max_seq_length:
        if len(tokens) % max_seq_length != 0:
            max_seq_length //= 2

        for i in range(0, len(tokens), len_text):
            slide = 0 if i == 0 else overlap
            truncated_tokens = tokens[i - slide:i + len_text]
            texts.append(' '.join(truncated_tokens))

        try:
            if len(texts) > 1 and len(tokenize_words(texts[-1])) + len(tokenize_words(texts[-2])) < len_text:
                texts[-2] = texts[-2] + texts[-1]
                texts.pop()
        except IndexError:
            print("\ntext: ", text)
            print(len(bert_tokens))
            print("len_text: ", len_text)
            print("tokens_len: ", len(tokens))
    else:
        texts.append(text)

    return texts

def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = ' '.join(word for word in wordpunct_tokenize(text)
                    if word not in string.punctuation)
    return text
def preprocess_text(text, split_mode='max_len', max_len=512, overlap=20):
    """
    Preprocess text for prediction
    :param overlap:  overlap between sentences
    :param max_len:  max length of sentence
    :param split_mode:  mode of splitting text
    :param text: text to preprocess
    :return:  list of preprocessed text
    """
    text = remove_extra_punctuation_regex(text)
    text = add_space_after_punkt(text)
    paragraphs = text.split('\n')
    if split_mode == 'sentence':
        paragraphs = split_in_sentences(text)
    elif split_mode == 'max_len':
        if max_len and overlap:
            paragraphs = truncate_texts_by_max_len(text, max_len, overlap)
        else:
            raise ValueError('Max length and overlap cannot be None when spliting by max length.')

    return list(map(lambda x: remove_punctuation(x).lower(), paragraphs))


def transform_sentences(text_, groups):
    new_text_list = list(text_)

    shift = 0

    for out in groups:
        punkt = '.' if out['entity_group'] == 'PERIOD' else ','
        if out['end'] + shift < len(new_text_list) - 1:
            new_text_list.insert(out['end'] + shift, punkt)
            shift += 1

    return ''.join(new_text_list)
def get_predicted_labels(model, sentence: str):
    predicted_labels = model.predict([sentence], )[0]

    y_pred = []
    for i, pred in predicted_labels:
        y_pred.append(list(map(lambda item: list(item.values())[0], pred)))
    return y_pred
