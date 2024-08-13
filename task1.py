import math as m
import multiprocessing
import os
import string
import time
import json
import random as rd
from collections import Counter
from multiprocessing import Pool

import nltk
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')

# initializing global values
stopwords_en = set(stopwords.words('english'))
stopwords_en.add('the')  # "the" is not included in the downloaded english stopword corpus, I decided to add it in.
lemmatizer = WordNetLemmatizer()
# please make sure the collection 'gov' is in the same directory as this file, or replace doc_path with the path to
# 'gov\\documents'
doc_path = os.path.join(os.getcwd(), 'gov\\documents')
# doc_cap is the number of folders in 'gov\\documents' that the script takes as the total collection. My pc can only
# handle upto a maximum of 40 before encountering MemoryError. Set the value to len(os.listdir(doc_path)) if such
# limitations do not apply.
doc_cap = 40
SAMPLING_PERCENTAGE = 10000000/58039392 # My uid is u8039392
PROCESSOR_COUNT = 12 # My cpu has 6 physical cores or 12 logical processors

# Naming method: listof_elem_elemofelem_... So listof_file_wordtokens is a list of files, which are lists of word
# tokens.
listof_file_wtokens = []
listof_wtokens = []  # Important! listof_wtokens is built from listof_file_wtokens after stopword removal.
listof_file_stokens = []
listof_file_wstuples = []
listof_file_sent_postags = []
listof_file_postags = []
listof_postags = []   # Important! listof_postags is built from listof_file_postags after stopword removal.
listof_file_lemmas = []
listof_file_2grams = []
listof_file_3grams = []
listof_file_nouns = []
listof_nouns = []
listof_file_verbs = []
listof_verbs = []
listof_file_adjs = []
listof_adjs = []
listof_file_advs = []
listof_advs = []

# Collections to be built and used for analysis
# dictionary of word-frequency pair
lexicon_tks = dict()
# dictionary of lemma-frequency pair
lexicon_lms = dict()

# Suite of document-level statistics. Unless otherwise stated, statistics are taken after stopwords removal.
# Initialized here solely for clarity and organization.
# list of total number of word tokens (in a file)
listof_tkcount = []
# total number of word tokens in document
total_tks = None
# minimum number of tokens (in a file)
min_tks = None
# maximum number of tokens (in a file)
max_tks = None
# mean number of tokens (in a file)
mean_tks = None
# number of unique tokens in document
total_uniq_tks = None
# list of total number of lemmas (in a file)
listof_lmcount = []
# total number of lemmas in document
total_lms = None
# minimum number of lemmas (in a file)
min_lms = None
# maximum number of lemmas (in a file)
max_lms = None
# mean number of lemmas (in a file)
mean_lms = None
# number of unique lemmas in document
total_uniq_lms = None

# Suite of linguistic characteristics
# list of most common unigrams (in a file) and their frequencies
listof_mc1g = []
# most common ten unigrams in document and its frequency
mc_ten_1gram = []
# list of most common bigrams (in a file) and their frequencies
listof_mc2g = []
# most common ten bigrams in document and its frequency
mc_ten_2gram = []
# list of most common trigrams (in a file) and their frequencies
listof_mc3g = []
# most common ten trigrams in document and its frequency
mc_ten_3gram = []
# list of most common nouns (in a file) and their frequencies
listof_mcN = []
# most common ten nouns in document and its frequency
mc_ten_noun = []
# list of most common verbs (in a file) and their frequencies
listof_mcV = []
# most common ten verbs in document and its frequency
mc_ten_verb = []
# list of most common adjectives (in a file) and their frequencies
listof_mcJ = []
# most common ten adjectives in document and its frequency
mc_ten_adj = []
# list of most common adverbs (in a file) and their frequencies
listof_mcR = []
# most common ten adverbs in document and its frequency
mc_ten_adv = []


def tknz_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
        return word_tokenize(text), sent_tokenize(text)


# returns a list of list of word tokens
def tknz_folder(folder_path):
    return [tknz_file(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)]


# takes as argument a list (file) of strings (sentences), return a list (file) of a list (sentence) of strings (
# words) from each sentence.
def pos_tag(slist):
    return nltk.pos_tag_sents([nltk.word_tokenize(s) for s in slist])


# lambda function for generating valid tag input for lemmatizer.lemmatize()
wordnet_tag = lambda t: 'a' if t == 'j' else (t if t in ['n', 'v', 'r'] else 'n')

# tokenize -> -> POS tagging -> stopword removal -> lemmatisation
if __name__ == '__main__':
    start = time.time()
    with Pool(PROCESSOR_COUNT) as p:
        folder_path_list = [os.path.join(doc_path, foldername) for foldername in os.listdir(doc_path)[:doc_cap]]
        for t in p.map(tknz_folder, folder_path_list):
            listof_file_wstuples.extend(t)
    print('%s files processed from the collection' % len(listof_file_wstuples))
    finish_tknz = time.time()
    print('Tokenization finished in %s seconds' % (finish_tknz - start))

    listof_file_wtokens[:] = [t[0] for t in listof_file_wstuples]
    listof_file_stokens[:] = [t[1] for t in listof_file_wstuples]

    # POS-tagging, before stopword removal

    # Unfortunately MemoryError is encountered when using multiple threads for POS-tagging the entire collection.
    # This error is not encountered with smaller collection sizes, such as the first 12 folders.
    with Pool(PROCESSOR_COUNT) as p:
      listof_file_sent_postags[:] = p.map(pos_tag, listof_file_stokens)

    # Unfortunately iterating through every file in the collection also cause the program to halt at around 20,000 files.
    # for i in range(len(listof_file_stokens)):
    #     listof_file_sent_postags.append(pos_tag(listof_file_stokens[i]))

    finish_postag = time.time()
    print('POS-tagging finished in %s seconds' % (finish_postag - finish_tknz))

    # stopword removal. Note that stopwords are detected with stopwords_en while ignoring cases.
    listof_file_wtokens[:] = [[w for w in x if w.lower() not in string.punctuation and w.lower() not in stopwords_en] for x in
                              listof_file_wtokens]
    listof_file_sent_postags[:] = [
        [[t for t in s if str(t[0]).lower() not in string.punctuation and str(t[0]).lower() not in stopwords_en] for s in f] for f in
        listof_file_sent_postags]
    listof_file_postags[:] = [[t for s in f for t in s] for f in listof_file_sent_postags]
    finish_swr = time.time()
    print('Stopwords removal finished in %s seconds' % (finish_swr - finish_postag))

    listof_wtokens[:] = [tk for f in listof_file_wtokens for tk in f]
    listof_postags[:] = [pt for f in listof_file_postags for pt in f]

    # build lexicon_tks
    for f in listof_file_wtokens:
        for w in f:
            if w in lexicon_tks:
                lexicon_tks[w] += 1
            else:
                lexicon_tks[w] = 1

    # lemmatisation; nltk's WolrdNetLemmatizer is case sensitive, so each word is changed to lowercases.
    listof_file_lemmas = [[lemmatizer.lemmatize(w.lower(), pos=wordnet_tag(t[0].lower())) for (w, t) in x] for x in
                          listof_file_postags]
    finish_lmtz = time.time()
    print('Lemmatisation finished in %s seconds' % (finish_lmtz - finish_swr))
    print('Preprocessing completed, %s seconds used.' % (finish_lmtz - start))

    # build lexicon_lms
    for f in listof_file_lemmas:
        for lm in f:
            if lm in lexicon_lms:
                lexicon_lms[lm] += 1
            else:
                lexicon_lms[lm] = 1

    # calculate collection-level statistics:
    # list of total number of word tokens (in a file)
    listof_tkcount[:] = [len(f) for f in listof_file_wtokens]
    # total number of word tokens in document
    total_tks = np.sum(listof_tkcount)
    # minimum number of tokens (in a file)
    min_tks = min(listof_tkcount) if len(listof_tkcount) > 0 else 0
    # maximum number of tokens (in a file)
    max_tks = max(listof_tkcount) if len(listof_tkcount) > 0 else 0
    # mean number of tokens (in a file)
    mean_tks = np.mean(listof_tkcount) if len(listof_tkcount) > 0 else 0
    # number of unique tokens in document
    total_uniq_tks = len(lexicon_tks)
    # list of total number of lemmas (in a file)
    listof_lmcount[:] = [len(f) for f in listof_file_lemmas]
    # total number of lemmas in document
    total_lms = np.sum(listof_lmcount)
    # minimum number of lemmas (in a file)
    min_lms = min(listof_lmcount) if len(listof_lmcount) > 0 else 0
    # maximum number of lemmas (in a file)
    max_lms = max(listof_lmcount) if len(listof_lmcount) > 0 else 0
    # mean number of lemmas (in a file)
    mean_lms = np.mean(listof_lmcount) if len(listof_lmcount) > 0 else 0
    # number of unique lemmas in document
    total_uniq_lms = len(lexicon_lms)

    # linguistic characteristics
    # list of most common unigrams (in a file). Ignores separation by punctuation and sentences.
    listof_mc1g = [(Counter(ngrams(f, 1)).most_common(1))[0] if len(f) >= 1 else None for f in listof_file_wtokens]
    # most common ten unigrams in document. Ignores separation by punctuation, sentences and files.
    mc_ten_1gram = Counter(ngrams(listof_wtokens, 1)).most_common(10 if len(listof_wtokens) >= 10 else len(listof_wtokens))
    # list of most common bigrams (in a file)
    listof_mc2g = [(Counter(ngrams(f, 2)).most_common(1))[0] if len(f) >= 2 else None for f in listof_file_wtokens]
    # most common ten bigrams in document.
    mc_ten_2gram = Counter(ngrams(listof_wtokens, 2)).most_common(10 if len(listof_wtokens) >= 11 else len(listof_wtokens)-1)
    # list of most common trigrams (in a file)
    listof_mc3g = [(Counter(ngrams(f, 3)).most_common(1))[0] if len(f) >= 3 else None for f in listof_file_wtokens]
    # most common ten trigrams in document.
    mc_ten_3gram = Counter(ngrams(listof_wtokens, 3)).most_common(10 if len(listof_wtokens) >= 12 else len(listof_wtokens)-2)

    # build lists of nouns, verbs, adjectives and adverbs
    for i in range(len(listof_file_postags)):
        listof_file_nouns.append([])
        listof_file_verbs.append([])
        listof_file_adjs.append([])
        listof_file_advs.append([])
        for tup in listof_file_postags[i]:
            tag = wordnet_tag(tup[1][0].lower())
            if tag == 'n':
                listof_file_nouns[i].append(tup[0])
            elif tag == 'v':
                listof_file_verbs[i].append(tup[0])
            elif tag == 'a':
                listof_file_adjs[i].append(tup[0])
            elif tag == 'r':
                listof_file_advs[i].append(tup[0])

    listof_nouns = [w for f in listof_file_nouns for w in f]
    listof_verbs = [w for f in listof_file_verbs for w in f]
    listof_adjs = [w for f in listof_file_adjs for w in f]
    listof_advs = [w for f in listof_file_advs for w in f]
    # most common ten nouns in document
    mc_ten_noun = Counter(listof_nouns).most_common(10 if len(listof_nouns) >= 10 else len(listof_nouns))
    # most common ten verbs in document
    mc_ten_verb = Counter(listof_verbs).most_common(10 if len(listof_verbs) >= 10 else len(listof_verbs))
    # most common ten adjectives in document
    mc_ten_adj = Counter(listof_adjs).most_common(10 if len(listof_adjs) >= 10 else len(listof_adjs))
    # most common ten adverbs in document
    mc_ten_adv = Counter(listof_advs).most_common(10 if len(listof_advs) >= 10 else len(listof_advs))
    # list of most common nouns (in a file)
    listof_mcN = [Counter(f).most_common(1) if len(f) > 0 else None for f in listof_file_nouns]
    # list of most common verbs (in a file)
    listof_mcV = [Counter(f).most_common(1) if len(f) > 0 else None for f in listof_file_verbs]
    # list of most common adjectives (in a file)
    listof_mcJ = [Counter(f).most_common(1) if len(f) > 0 else None for f in listof_file_adjs]
    # list of most common adverbs (in a file)
    listof_mcR = [Counter(f).most_common(1) if len(f) > 0 else None for f in listof_file_advs]

    # Display collection-level analysis
    print('\n\n' + '==============================================================================================' + '\n'
          + 'Collection-Level Statistics (after stopword removal): \n'
          + ('  Total number of tokens in collection: %s' % total_tks) + '\n'
          + ('  Minimum number of tokens in a file: %s' % min_tks) + '\n'
          + ('  Maximum number of tokens in a file: %s' % max_tks) + '\n'
          + ('  Mean number of tokens in a file: %s' % mean_tks) + '\n'
          + ('  Number of unique tokens in collection: %s' % total_uniq_tks) + '\n\n'
          + ('  Total number of lemmas in collection: %s' % total_lms) + '\n'
          + ('  Minimum number of lemmas in a file: %s' % min_lms) + '\n'
          + ('  Mazimum number of lemmas in a file: %s' % max_lms) + '\n'
          + ('  Number of unique lemmas in collection: %s' % total_uniq_lms) + '\n\n'
          + ('  Top 10 most common unigrams: ' + str(mc_ten_1gram)) + '\n'
          + ('  Top 10 most common bigrams: ' + str(mc_ten_2gram)) + '\n'
          + ('  Top 10 most common trigrams: ' + str(mc_ten_3gram)) + '\n\n'
          + ('  Top 10 most common nouns: ' + str(mc_ten_noun)) + '\n'
          + ('  Top 10 most common verbs: ' + str(mc_ten_verb)) + '\n'
          + ('  Top 10 most common adjectives: ' + str(mc_ten_adj)) + '\n'
          + ('  Top 10 most common adverbs: ' + str(mc_ten_adv)) + '\n'
          + '==============================================================================================')


    # Apply Zipf's law and perform sampling
    listof_folder_paths = [os.path.join(doc_path, folder) for folder in os.listdir(doc_path)]
    listof_file_paths = [os.path.join(folder_path, file) for folder_path in listof_folder_paths[:doc_cap] for file in os.listdir(folder_path)]
    listof_sample_index = rd.sample(range(len(listof_file_paths)), m.ceil(SAMPLING_PERCENTAGE * len(listof_file_paths)))

    # Generate sampled data collections
    sample_listof_file_wtokens = [listof_file_wtokens[i] for i in listof_sample_index]
    sample_listof_wtokens = [w for f in sample_listof_file_wtokens for w in f]
    sample_lexicon_tks = dict()
    for f in sample_listof_file_wtokens:
        for w in f:
            if w in sample_lexicon_tks:
                sample_lexicon_tks[w] += 1
            else:
                sample_lexicon_tks[w] = 1
    sample_listof_file_lemmas = [listof_file_lemmas[i] for i in listof_sample_index]
    sample_lexicon_lms = dict()
    for f in sample_listof_file_lemmas:
        for lm in f:
            if lm in sample_lexicon_lms:
                sample_lexicon_lms[lm] += 1
            else:
                sample_lexicon_lms[lm] = 1
    sample_listof_file_nouns = [listof_file_nouns[i] for i in listof_sample_index]
    sample_listof_file_verbs = [listof_file_verbs[i] for i in listof_sample_index]
    sample_listof_file_adjs = [listof_file_adjs[i] for i in listof_sample_index]
    sample_listof_file_advs = [listof_file_advs[i] for i in listof_sample_index]
    sample_listof_nouns = [w for f in sample_listof_file_nouns for w in f]
    sample_listof_verbs = [w for f in sample_listof_file_verbs for w in f]
    sample_listof_adjs = [w for f in sample_listof_file_adjs for w in f]
    sample_listof_advs = [w for f in sample_listof_file_advs for w in f]


    # Acquire sample-level statistics
    sample_listof_tkcount = [len(listof_file_wtokens[i]) for i in listof_sample_index]
    sample_total_tks = np.sum(sample_listof_tkcount)
    sample_min_tks = min(sample_listof_tkcount) if len(sample_listof_tkcount) > 0 else 0
    sample_max_tks = max(sample_listof_tkcount) if len(sample_listof_tkcount) > 0 else 0
    sample_mean_tks = np.mean(sample_listof_tkcount) if len(sample_listof_tkcount) > 0 else 0
    sample_total_uniq_tks = len(sample_lexicon_tks)
    sample_listof_lmcount = [len(listof_file_lemmas[i]) for i  in listof_sample_index]
    sample_total_lms = np.sum(sample_listof_lmcount)
    sample_min_lms = min(sample_listof_lmcount) if len(sample_listof_lmcount) > 0 else 0
    sample_max_lms = max(sample_listof_lmcount) if len(sample_listof_lmcount) > 0 else 0
    sample_mean_lms = np.mean(sample_listof_lmcount) if len(sample_listof_lmcount) > 0 else 0
    sample_total_uniq_lms = len(lexicon_lms)

    # Acquire sample linguistic characteeristics
    sample_mc_ten_1gram = Counter(ngrams(sample_listof_wtokens, 1)).most_common(10 if len(sample_listof_wtokens) >= 10 else len(sample_listof_wtokens))
    sample_mc_ten_2gram = Counter(ngrams(sample_listof_wtokens, 2)).most_common(10 if len(sample_listof_wtokens) >= 11 else len(sample_listof_wtokens)-1)
    sample_mc_ten_3gram = Counter(ngrams(sample_listof_wtokens, 3)).most_common(10 if len(sample_listof_wtokens) >= 12 else len(sample_listof_wtokens)-2)
    sample_mc_ten_noun = Counter(sample_listof_nouns).most_common(10 if len(sample_listof_nouns) >= 10 else len(sample_listof_nouns))
    sample_mc_ten_verb = Counter(sample_listof_verbs).most_common(10 if len(sample_listof_verbs) >= 10 else len(sample_listof_verbs))
    sample_mc_ten_adj = Counter(sample_listof_adjs).most_common(10 if len(sample_listof_adjs) >= 10 else len(sample_listof_adjs))
    sample_mc_ten_adv = Counter(sample_listof_advs).most_common(10 if len(sample_listof_advs) >= 10 else len(sample_listof_advs))

    # Display sample-level analysis
    print('\n\n' + '==============================================================================================' + '\n'
          + 'Sample-Level Statistics (after stopword removal): \n'
          + ('  Total number of tokens in sampled collection: %s' % sample_total_tks) + '\n'
          + ('  Minimum number of tokens in a file: %s' % sample_min_tks) + '\n'
          + ('  Maximum number of tokens in a file: %s' % sample_max_tks) + '\n'
          + ('  Mean number of tokens in a file: %s' % sample_mean_tks) + '\n'
          + ('  Number of unique tokens in collection: %s' % sample_total_uniq_tks) + '\n\n'
          + ('  Total number of lemmas in collection: %s' % sample_total_lms) + '\n'
          + ('  Minimum number of lemmas in a file: %s' % sample_min_lms) + '\n'
          + ('  Mazimum number of lemmas in a file: %s' % sample_max_lms) + '\n'
          + ('  Number of unique lemmas in sampled collection: %s' % sample_total_uniq_lms) + '\n\n'
          + ('  Top 10 most common unigrams: ' + str(sample_mc_ten_1gram)) + '\n'
          + ('  Top 10 most common bigrams: ' + str(sample_mc_ten_2gram)) + '\n'
          + ('  Top 10 most common trigrams: ' + str(sample_mc_ten_3gram)) + '\n\n'
          + ('  Top 10 most common nouns: ' + str(sample_mc_ten_noun)) + '\n'
          + ('  Top 10 most common verbs: ' + str(sample_mc_ten_verb)) + '\n'
          + ('  Top 10 most common adjectives: ' + str(sample_mc_ten_adj)) + '\n'
          + ('  Top 10 most common adverbs: ' + str(sample_mc_ten_adv)) + '\n'
          + '==============================================================================================')

    # File statistics are stored in the following variables. Due to time and scope limitations I chose not to display
    # them, but they can be accessed here for comparison with document and sample level statistics, and manipulated
    # into dataframes and .csv files.
    # print(listof_tkcount, listof_lmcount)
    # print(listof_mc1g, listof_mc2g, listof_mc3g, listof_mcN, listof_mcV, listof_mcJ, listof_mcR)