import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
import pandas as pd
import deepsmiles as deeps
from Bio.ExPASy import Prosite,Prodoc


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ##

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                         "t": 61, "y": 62, "|": 63, ":": 64, ",": 65, "*": 66}

CHARCANsmilen = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27,
                                "u": 63, "t": 28, "y": 64, "|": 65, ":": 66,
                                ",": 67, "*": 68}

CHARISOsmilen = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ##


def label_smiles(line, max_smilen, smi_ch_ind):
	X = np.zeros(max_smilen)
	for i, ch in enumerate(line[:max_smilen]):
		X[i] = smi_ch_ind[ch]

	return X


def label_sequence(line, max_seqlen, smi_ch_ind):
	X = np.zeros(max_seqlen)

	for i, ch in enumerate(line[:max_seqlen]):
		X[i] = smi_ch_ind[ch]

	return X


def prosite_parser():
    """
    Converts Prosite patterns in strings parseable as regex
    """
    pattern_replacements = {'-' : '',
                        '{' : '[^', # {X} = [^X]
                        '}' : ']',
                        '(' : '{', # (from, to) = {from, to}
                        ')' : '}',
                        'X' : '.', # x, X = any (.)
                        'x' : '.',
                        '<' : '^', # < = N-terminal
                        '>' : '$' # > = C-terminal
                        }
    patterns = {}
    names = []
    nameset = set()
    with open("data/prosite.dat", "r") as handle:
        records = Prosite.parse(handle)
        for record in records:
            pattern = record.pattern.strip('.')
            # Transform ProSite patterns
            # to regular expressions readable by re module
            for pat, repl in pattern_replacements.items():
                pattern = pattern.replace(pat, repl)
            patterns[record.name] = pattern
            names.append(record.name)
            nameset.add(record.name)
    assert(len(names) == len(nameset))
    return patterns


def extract_domains(FLAGS, seqs):
    """
    Return matrix with Prosite domains from input sequences
    """
    Xdoms = []
    patterns = prosite_parser() # dict with domain patterns
    FLAGS.domset_size = len(patterns.keys()) + 1
    maxlen = 0
    for seq in seqs:
        seq = str(seq)
        hits = []
        for idx, pattern in enumerate(patterns.values()):
            if pattern != "" and re.search(pattern, seq):
                hits.append(idx + 1)
        Xdoms.append(hits)
        maxlen = max(maxlen, len(hits))
    FLAGS.max_dom_len = maxlen
    # Pad to max number of domains per sequence
    for idx in range(len(Xdoms)):
        length = len(Xdoms[idx])
        Xdoms[idx].extend(np.zeros(maxlen-length))
        assert(len(Xdoms[idx]) == maxlen)
    print(str(len(seqs)) + " > " + str(len(Xdoms)))
    return Xdoms


def deepsmiles(smiles):
    """Build DeepSMILES representation of given SMILES vector

    :smiles: smiles vector
    :returns: DeepSMILES representation

    """
    converter = deeps.Converter(rings=True, branches=True)
    dsmiles = []
    for smi in smiles:
        dsmiles.append(converter.encode(smi))

    return dsmiles


def char_representation(data, max_len, char_dict):
    X = []
    for d in data:
        d = str(d).replace(" ", "")[:max_len]
        x = np.zeros(max_len)
        for i, ch in enumerate(d):
            print(ch)
            x[i] = char_dict[ch]
        X.append(x)

    return X


def build_wordict(data, max_len, wordlen):
    """Build word representation for protein sequences

    :X: dictionary of protein sequences
    :max_len: max length or word representation
    :wordlen: length of moving wordlen to create word "alphabet"
    :returns: word dictionary (word : int), word_representation of data

    """
    word_set = set()
    for d in data:
        d = str(d).strip(" ")
        length = min(len(d), max_len) - wordlen
        for idx in range(length):
            word = d[idx : idx+wordlen]
            assert(len(word) == wordlen)
            word_set.add(word)
    word_dict = dict(zip(word_set,
                         [i for i in range(1, len(word_set)+1)]))

    return word_dict


def word_representation(word_dict, data, max_len, wordlen):
    X = []

    for d in data:
        d = str(d).strip(" ")
        x = np.zeros(max_len)
        length = min(len(d), max_len) - wordlen
        for idx in range(length):
            word = d[idx : idx+wordlen]
            x[idx] = word_dict[word]
        assert(len(x)==max_len)
        X.append(x)

    return X


## ######################## ##
#
#  DATASET Class
#
## ######################## ##

class DataSet(object):
    def __init__(self, path, seqlen, smilen, word_representation,
               seq_wordlen, smi_wordlen, need_shuffle = False):
        self.path = path
        self.seqlen = seqlen
        self.smilen = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET
        self.charsmiset_size = CHARISOsmilen

        # Word representation: wideDTA
        self.word_representation = word_representation
        self.seq_wordlen = seq_wordlen
        self.smi_wordlen = smi_wordlen
        self.smi_words = dict
        self.seq_words = dict



    def read_sets(self, FLAGS):
# path should be the dataset folder kiba/ or bindingDB/
        path = self.path
        print("Reading %s start" % path)

        train = json.load(open(path + "train.txt"))
        test = json.load(open(path + "test.txt"))

        return train, test



    def parse_kiba(self, FLAGS):
        path = self.path
        print("Read %s start" % path)

        ligands = json.load(open(path+"ligands_can.txt"),
                object_pairs_hook=OrderedDict).values()
        proteins = json.load(open(path+"proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(path + "Y","rb"), encoding='latin1')
        if FLAGS.is_log:
            Y = -(np.log10(Y/(math.pow(10,9))))

        if FLAGS.deep_smiles:
            ligands = deepsmiles(ligands)

        if self.word_representation:
            smi_words = build_wordict(ligands, self.smilen,
                                          self.smi_wordlen)
            XD = word_representation(smi_words, ligands, self.smilen,
                                     self.smi_wordlen)

            FLAGS.smi_wordset_size = len(smi_words.keys())
            self.smi_words = smi_words
            print("Number of unique SMILES words: " +
                    str(FLAGS.smi_wordset_size))

            seq_words = build_wordict(proteins.values(), self.seqlen,
                                                 self.seq_wordlen)
            XT = word_representation(seq_words, proteins.values(), self.seqlen,
                                                 self.seq_wordlen)

            FLAGS.seq_wordset_size = len(seq_words.keys())
            self.seq_words = seq_words
            print("Number of unique sequence words: " +
                    str(FLAGS.seq_wordset_size))

        else:
            XD = []
            XT = []
            for d in ligands.keys():
                XD.append(label_smiles(ligands[d], self.smilen, self.charsmiset))

            for t in proteins.keys():
                XT.append(label_sequence(proteins[t], self.seqlen, self.charseqset))

        if FLAGS.extract_domains:
            if not FLAGS.provided_domains:
                Xdoms = extract_domains(FLAGS, list(proteins.values()))
                try:
                    with open(self.path + PID + "-domains.txt", "w") as f:
                        f.write(str(list(Xdoms)))
                except:
                    print(Xdoms)
                    print("\nIssue saving domains\n")
            else:
                with open(self.path + "domains.txt") as ds:
                    Xdoms = eval(str(ds.read()))
                FLAGS.max_dom_len = len(Xdoms[0])
                domset_size = 0
                for doms in Xdoms:
                    domset_size = max(domset_size, max(doms))
                FLAGS.domset_size = domset_size
        else:
            Xdoms = []

        return XD, XT, Xdoms, Y



    def parse_data(self,FLAGS):
        """"""
        trainingpath = self.path + "IC50_training.csv"
        training = pd.read_csv(trainingpath, sep=",")

        testpath = self.path + "IC50_test.csv"
        test = pd.read_csv(testpath, sep=",")

        trainsmi = training.smiles
        trainseq = training.seq
        Ytrain = training.affinity

        testsmi = test.smiles
        testseq = test.seq
        Ytest = test.affinity

        if FLAGS.deep_smiles:
            trainsmi = deepsmiles(trainsmi)
            testsmi = deepsmiles(testsmi)

        smi = list(trainsmi)+list(testsmi)
        seq = list(trainseq)+list(testseq)


        print("\nUnique sequences: " + str(len(pd.unique(seq))))
        print("Unique SMILES: " + str(len(pd.unique(smi))))


        if FLAGS.word_representation:

            # Build word dictionaries
            smi_words = build_wordict(smi, self.smilen, self.smi_wordlen)
            seq_words = build_wordict(seq, self.seqlen, self.seq_wordlen)

            FLAGS.smi_wordset_size = len(smi_words.keys())
            self.smi_words = smi_words
            print("Number of unique SMILES words: " +
                    str(FLAGS.smi_wordset_size))

            FLAGS.seq_wordset_size = len(seq_words.keys())
            self.seq_words = seq_words
            print("Number of unique sequence words: " +
                    str(FLAGS.seq_wordset_size))

            # Build word representations
            XDtrain = word_representation(smi_words, trainsmi, self.smilen,
                                          self.smi_wordlen)

            XTtrain = word_representation(seq_words, trainseq, self.seqlen,
                                                 self.seq_wordlen)


            XDtest = word_representation(smi_words, testsmi, self.smilen,
                                          self.smi_wordlen)

            XTtest = word_representation(seq_words, testseq, self.seqlen,
                                                 self.seq_wordlen)
        else:
            XDtrain = char_representation(trainsmi, self.smilen,
                                          self.charsmiset)
            XTtrain = char_representation(trainseq, self.seqlen,
                                          self.charseqset)

            XDtest = char_representation(testsmi, self.smilen, self.charsmiset)
            XTtest = char_representation(testseq, self.seqlen, self.charseqset)


        if not FLAGS.provided_domains:
            Xdoms = extract_domains(FLAGS, seq)
            try:
                with open(self.path + PID + "-domains.txt", "w") as f:
                    f.write(str(Xdoms))
            except:
                print("\nIssue saving domains\n")
        else:
            print("\nUsing externally provided domains...\n")
            with open(self.path + "domains.txt") as ds:
                Xdoms = eval(str(ds.read()))
            FLAGS.max_dom_len = len(Xdoms[0])
            domset_size = 0
            for doms in Xdoms:
                domset_size = max(domset_size, max(doms))
            FLAGS.domset_size = domset_size

        Xdomtrain = Xdoms[:len(trainseq)]
        Xdomtest = Xdoms[len(trainseq):]
        print("Max number of domains per protein: " + str(FLAGS.max_dom_len))
        assert(len(Xdomtrain)==len(XTtrain))
        assert(len(Xdomtest)==len(XTtest))

        return XDtrain, XTtrain, Xdomtrain, Ytrain, XDtest, XTtest, Xdomtest, Ytest
