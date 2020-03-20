# Driver code to train and test English-Spanish translator using monolingual data.
# Approach inspired by "Towards Neural Machine Translation with Partially Aligned Corpora"
# link to paper here: https://www.aclweb.org/anthology/I17-1039/
# neural network architecture adopted from PyTorch seq2seq tutorial

from __future__ import unicode_literals, print_function, division

from lang import *

from io import open
import unicodedata
import string
import re
import random

from nltk.translate.api import AlignedSent, Alignment
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.phrase_based import phrase_extraction

from collections import defaultdict

from encoder import *
from decoder import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MIN_PHRASE_LEN = 4

# recycled from other tasks
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^\w\d'\s]+", r'', s)
    return s

# read in parallel sentence pairs, to be used for phrase table
def createPairs(lang1, lang2):
    lines_src = open('data/dev.%s' % (lang1), encoding='utf-8').\
        read().strip().split('\n')

    lines_trg = open('data/dev.%s' % (lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(lines_src[i]), normalizeString(lines_trg[i])] for i in range(len(lines_src))]

    return pairs

pairs = createPairs('en', 'es')
print(random.choice(pairs))

# create tokenized bitext to prepare for IBM alignment
def createBitext(pairs):
    bitext = []
    for pair in pairs:
        src = pair[0].split()
        trg = pair[1].split()
        bitext.append(AlignedSent(src, trg))
    return bitext

bitext = createBitext(pairs)


ibm = IBMModel2(bitext, 10)
test_sentence = random.choice(bitext)
#print(test_sentence.words)
#print(test_sentence.mots)
#print(tuple(test_sentence.alignment))

for i in range(len(bitext)):
	#print(bitext[i].alignment)
	newAlignment = []
	for item in tuple(bitext[i].alignment):

        # only keep word pairings where neither of the words is None
		if None not in item:
			newAlignment.append(item)
	bitext[i].alignment = Alignment(newAlignment)


all_phrases = []
for pair in bitext:

	srctext = ' '.join(word for word in pair.words)
	trgtext = ' '.join(word for word in pair.mots)
	alignment = tuple(pair.alignment)

	phrases = phrase_extraction(srctext, trgtext, alignment)
	for phrase in phrases:
		all_phrases.append(phrase)

# build dict matching english phrases to spanish phrases
phrase_occ = {}
for row in all_phrases:
    src = row[2]
    trg = row[3]
    if src not in phrase_occ:
        translations = defaultdict()
        translations[trg] = 1
        phrase_occ[src] = translations
    elif trg not in phrase_occ[src]:
        phrase_occ[src][trg] = 1
    else:
        phrase_occ[src][trg] += 1

# calculate probabilites of pairings
probs = {}
for src in phrase_occ:
    total_trans = 0
    for trg in phrase_occ[src]:
        total_trans += phrase_occ[src][trg]

    for trg in phrase_occ[src]:
        pair = (src, trg)
        probs[pair] = float(phrase_occ[src][trg]/total_trans)

# keep single words with higher than 0.5 probability
singletons = []
for pair in probs:
    if probs[pair] > 0.5 and len(pair[0].split()) == 1 and len(pair[1].split()) == 1:
        singletons.append(pair)

# only keep the pairings with 0.5 probability and longer than min phrase length
keep = []
for pair in probs:
    if probs[pair] > 0.5 and len(pair[0].split()) >= MIN_PHRASE_LEN and len(pair[1].split()) >= MIN_PHRASE_LEN:
        keep.append(pair)

# create fast hashset lookup tables
single_en_lookup = set([singletons[i][0] for i in range(len(singletons))])
single_es_lookup = set([singletons[i][1] for i in range(len(singletons))])

en_lookup = set([keep[i][0] for i in range(len(keep))])
es_lookup = set([keep[i][1] for i in range(len(keep))])

# create pairings and Lang objects for monolingual corpora
def readMonoCorpora(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines_src = open('data/%s_unaligned' % (lang1), encoding='utf-8').\
        read().strip().split('\n')

    lines_trg = open('data/%s_unaligned' % (lang2), encoding='utf-8').\
        read().strip().split('\n')

    # make Lang instances
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    ua_pairs = []
    assert(len(lines_src) == len(lines_trg))

    for i in range(len(lines_src)):
        src_sent = normalizeString(lines_src[i])
        trg_sent = normalizeString(lines_trg[i])
        input_lang.addSentence(src_sent)
        output_lang.addSentence(trg_sent)
        ua_pairs.append([src_sent, trg_sent])
        #print("iteration", i)
        #print("English sentence:", src_sent)
        #print("Spanish sentence:", trg_sent)

    return input_lang, output_lang, ua_pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareMonoCorpora(lang1, lang2, reverse=False):
    input_lang, output_lang, ua_pairs = readMonoCorpora(lang1, lang2)

    ua_pairs = filterPairs(ua_pairs)
#    print("Counted words:")
#    print(input_lang.name, input_lang.n_words)
#    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, ua_pairs

input_lang, output_lang, ua_pairs = prepareMonoCorpora('en', 'es')

count = 0
for word in input_lang.word2index:
    #print(word)
    if word in en_lookup:
        count +=1 
print("num single words:", count)

pa_pairs = []

# algorithm to match parallel phrases
# iterate over every eligible phrase, check to see if it is contained in each sentence
# O(n eligible phrases * n aligned pairs)

# helper function: efficient way to check if phrase is contained in sentence
def phraseContained(A, B): 
    i, j = 0, 0 
    n, m = len(A), len(B)

    while (i < n and j < m): 
        if (A[i] == B[j]): 
            i += 1
            j += 1 

            if (j == m): 
                return True
          
        else: 
            i = i - j + 1
            j = 0
          
    return False

for word in keep:
    src_idx = None
    trg_idx = None

    for i in range(len(ua_pairs)):
        phrase0 = word[0].split()
        sentence0 = ua_pairs[i][0].split()
        if phraseContained(phrase0, sentence0):
            src_idx = i
            break

    for i in range(len(ua_pairs)):
        phrase1 = word[1].split()
        sentence1 = ua_pairs[i][1].split()
        if phraseContained(phrase1, sentence1):
            trg_idx = i
            break

    if src_idx is not None and trg_idx is not None:
        
        src_match = ua_pairs[src_idx]
        trg_match = ua_pairs[trg_idx]
        pa_pairs.append([src_match[0], trg_match[1]])

        if src_idx > trg_idx:
            ua_pairs.append([trg_match[0], src_match[1]])
            ua_pairs = ua_pairs[:trg_idx] + ua_pairs[trg_idx+1:src_idx] + ua_pairs[src_idx+1:]
            
        elif src_idx < trg_idx:
            ua_pairs.append([trg_match[0], src_match[1]])
            ua_pairs = ua_pairs[:src_idx] + ua_pairs[src_idx+1:trg_idx] + ua_pairs[trg_idx+1:]
            
        else:
            ua_pairs = ua_pairs[:src_idx] + ua_pairs[trg_idx+1:]

    #print("len of ua after", len(ua_pairs))
    #print("len of pa after", len(pa_pairs))
    #print("total_len", len(ua_pairs)+len(pa_pairs))

all_pairs = pa_pairs + ua_pairs
#print("len of all_pairs:", len(all_pairs))
#for row in all_pairs[:20]:
#    print("en", row[0])
#    print('es', row[1])

#print("len of pairs after alignment", len(all_pairs))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    #decoder_hidden = encoder_hidden
    decoder_hidden = encoder.initHidden()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=500, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(all_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(all_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 50000, print_every=500)
torch.save(encoder1, 'seq2seq-encoder_v1.pt')
torch.save(attn_decoder1, 'seq2seq-decoder_v1.pt')

evaluateRandomly(encoder1, attn_decoder1)
