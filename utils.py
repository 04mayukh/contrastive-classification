import pandas as pd
import numpy as np
import ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from collections import Counter
import emoji
import re
from sklearn.metrics import classification_report

def clear_m():
    gc.collect()
    torch.cuda.empty_cache()

def get_text_preprocessor():
    preprocessor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
          'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
          'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    return preprocessor

def get_data(path):
    df = pd.read_csv(path)
    text = df['text']
    labels = np.asarray(df['label'].astype(int))
    return text, labels


def sentiment_count(train_labels_string, emotion2idx, dataType):
    print("\n" + dataType)
    count = Counter(train_labels_string)
    sum_1 = 0
    for key, value in emotion2idx.items():
        print(key + ": ", count[value])
        sum_1 += count[value]
    print("Independent emotion: ", sum_1)
    

def print_text(texts,i,j):
    for u in range(i,j):
        print(texts[u])
        print()


def initialise_slang(path):
    f = open(path, "r")
    chat_words_str = f.read()
    chat_words_map_dict = {}
    chat_words_list = []

    for line in chat_words_str.split("\n"):
        if line != "":
            cw = line.split("=")[0]
            cw_expanded = line.split("=")[1]
            chat_words_list.append(cw)
            chat_words_map_dict[cw] = cw_expanded
    chat_words_list = set(chat_words_list)
    return chat_words_list, chat_words_map_dict


# Functions for chat word conversion
def chat_words_conversion(text, chat_words_list, chat_words_map_dict):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


# Function for removal of emoji
def convert_emojis(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub("_|-"," ",text)
    return text


def ekphrasis_pipe(sentence, text_processor):
    cleaned_sentence = " ".join(text_processor.pre_process_doc(sentence))
    return cleaned_sentence


def pre_process_text(text_, slang_path, text_processor):
    chat_words_list, chat_words_map_dict = initialise_slang(slang_path)
    text_ = text_.apply(lambda text: chat_words_conversion(text, chat_words_list, chat_words_map_dict))
    text_ = text_.apply(lambda text: convert_emojis(text))
    text_ = text_.apply(lambda text: ekphrasis_pipe(text, text_processor))
    return text_