#!/bin/python

# https://github.com/miotto/treetagger-python
# http://www.nltk.org/_modules/nltk/stem/rslp.html

import functools
import json
import re
import nltk
from types import SimpleNamespace
from treetagger import TreeTagger
from nltk.stem import RSLPStemmer
from unidecode import unidecode

COMMAND = SimpleNamespace(WEKA='weka', NORMALIZE='normalize')

NEGATIVE = 0
POSITIVE = 1

TEST_PERCENTAGE = 80
K = 100

LANGUAGE = "portuguese"

DEBUG = False


class NLTKHelper(object):
    def __init__(self, corpus_filename, command=None, param_file=None):
        assert corpus_filename
        assert command
        assert param_file

        self.corpus_filename = corpus_filename
        self.command = command
        self.param_filename = param_file
        self.corpus = None

        self.bow_positive = []
        self.bow_negative = []

        self.positive_tweets = []
        self.negative_tweets = []

        self.positive_flatened = []
        self.negative_flatened = []

        self.fdist_positive = []
        self.fdist_negative = []

        self._init_nltk()
        self._init_tree_tagger()

        if self.command == COMMAND.NORMALIZE:
            print("Normalizing...")
            self.pre_process()
        else:
            print("Running...")
            self.load_normalized()
            self.build_structure()

    def load_normalized(self):
        with open(self.param_filename, 'r') as f:
            norm = json.load(f)
            self.positive_tweets = norm['positive']
            self.negative_tweets = norm['negative']
            self.positive_flatened = [y for x in self.positive_tweets for y in x]
            self.negative_flatened = [y for x in self.negative_tweets for y in x]

    def _init_nltk(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('rslp')

        print("Initializing NLTK...")

        self.stopwords = nltk.corpus.stopwords.words(LANGUAGE)
        self.stemmer = RSLPStemmer()

    def _init_tree_tagger(self):
        self.tree_tagger = TreeTagger(language=LANGUAGE)

    @property
    def superlist(self):
        return self.positive_flatened + self.negative_flatened

    def build_structure(self):

        self.fdist_positive = nltk.FreqDist(self.positive_flatened)
        self.fdist_negative = nltk.FreqDist(self.negative_flatened)

        self.bow_positive = self.fdist_positive.most_common(K)
        self.bow_negative = self.fdist_negative.most_common(K)

        print("> Found {0} unique words <POSITIVE>".format(len(self.fdist_positive)))
        print("> Found {0} unique words <NEGATIVE>".format(len(self.fdist_negative)))
        print(40*"#")
        print("TOP {0} POSITIVE Words:".format(K))
        self.fdist_positive.tabulate(K)
        print(40 * "#")
        print("TOP {0} NEGATIVE Words:".format(K))
        self.fdist_negative.tabulate(K)
        print(40 * "#")

        self.make_weka_output()

        # # Get unique words
        # unique_words = set()
        # for norm in superlist:
        #     unique_words |= set(norm)

    def make_weka_output(self):
        _bow_positive = [t[0] for t in self.bow_positive]
        _bow_negative = [t[0] for t in self.bow_negative]

        # Get unique words
        superlist = set(_bow_positive) | set(_bow_negative)

        print("Mixed Bag-Of-Words length: {0}".format(len(superlist)))

        with open('out.arff', 'w+') as outfile:
            outfile.write("@relation notebookTweets\n")
            print("@relation notebookTweets")
            for att in superlist:
                outfile.write("@attribute '{0}' numeric\n".format(att))
                print("@attribute '{0}' numeric".format(att))

            outfile.write("@attribute 'class' {POSITIVE, NEGATIVE}\n\n")
            outfile.write("@data\n".format(att))
            print("@attribute class {POSITIVE, NEGATIVE}\n")
            print("@data")

            for tweet in self.positive_tweets:
                line = ""
                for unique in superlist:
                    if unique in tweet:
                        line += "1,"
                    else:
                        line += "0,"
                line += "POSITIVE"
                outfile.write(line + "\n")
                print(line)

            for tweet in self.negative_tweets:
                line = ""
                for unique in superlist:
                    if unique in tweet:
                        line += "1,"
                    else:
                        line += "0,"
                line += "NEGATIVE"
                outfile.write(line + "\n")
                print(line)

    def pre_process(self):
        positive, negative = self.cleanup()
        self.positive_tweets = self.normalize(positive)
        self.negative_tweets = self.normalize(negative)

        with open(self.param_filename, 'w+') as f:
            json.dump({'positive': self.positive_tweets, 'negative': self.negative_tweets}, f)

    def cleanup(self):
        positive = []
        negative = []
        with open(self.corpus_filename, 'r') as f:
            self.corpus = json.load(f)
            for idx, tweet in enumerate(self.corpus):
                # Removes neutral tweets
                if tweet['FINAL'] == '0':
                    continue

                # Removes consecutive characters as in:
                # Ooooooi => Oi
                # Undesired side-effect: Dell => Del
                # Whateverrrrr
                dedup = functools.partial(re.sub, r"(.)\1+", r"\1\1")

                t = tweet["Tweet"]

                if DEBUG:
                    print("#"*10)
                    print("BEFORE CLEANUP")
                    print(t)
                    print("#" * 5)

                # lowercase and removes final space
                t = t.lower().rstrip()

                # Removes tweet related crap, user refs and hashtags
                t = ' '.join(sw
                             for sw in t.split()
                             if not sw.startswith('http')
                             and not sw.startswith('@')
                             and not sw.startswith('#'))

                # Removing accents, such as in "é => e", makes us lose meaning
                # Removes all special characters
                t = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]+', '', t)

                # Dedup, as mentioned above
                t = dedup(t)

                if tweet["FINAL"] == '1':
                    positive.append(t)
                else:
                    negative.append(t)

                if DEBUG:
                    print("AFTER CLEANUP")
                    print(t)

        return positive, negative

    def get_part_of_speech(self, sentence):
        # https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/Portuguese-Tagset.html
        _pos = self.tree_tagger.tag(sentence)

        if DEBUG:
            print(_pos)

        _joined = []
        for _p in _pos:
            if _p[1][0] not in ['V', 'R', 'N', 'A']:
                continue
            # Infinitive
            _joined.append(_p[0]) if _p[2] == '<unknown>' else _joined.append(_p[2])
        return ' '.join(_joined)

    def normalize(self, corpus):
        normalized = []
        count = 1
        size = len(corpus)
        for sentence in corpus:
            # POS
            pos = self.get_part_of_speech(sentence)
            # Tokenize
            tokenized = self.tokenize(pos)
            # Stem
            stemmed = self.stem(tokenized)

            # Stopwords (b)
            final = self.remove_stopwords(stemmed)

            print("Normalized {0}/{1} tweets".format(count, size))
            count += 1
            normalized.append(final)

        return normalized

    @staticmethod
    def tokenize(sentence):
        # Tokenize
        t = nltk.word_tokenize(sentence, language=LANGUAGE)
        return t

    def stem(self, sentence):
        new = []
        for w in sentence:
            # Small orthographic fix
            if w == 'not':
                w = 'notebook'
            s = self.stemmer.stem(w)
            # Remove acentos
            new.append(unidecode(s))
        return new

    def remove_stopwords(self, sentence):
        new = []
        for w in sentence:
            if w not in self.stopwords:
                new.append(w)
        return new


def main(command='weka', file=None):

    print(command, file)

    fname = 'tweets.json'
    if DEBUG:
        fname = 'test.json'

    helper = NLTKHelper(corpus_filename=fname, command=command, param_file=file)
    if DEBUG:
        print(helper.positive_tweets)
        print(helper.negative_tweets)


