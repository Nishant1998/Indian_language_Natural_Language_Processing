#!/usr/bin/env python
# coding: utf-8

# In[16]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = ['unigram_char','bigram_char','trigram_char','quardrigram_char','unigram_word','bigram_word','trigram_word','unigram_syllable','bigram_syllable','trigram_syllable']

for i in file:
    name = i + ".csv"
    df = pd.read_csv(name)
    freq = df['frequency'].to_numpy()
    log_freq = np.log(freq)
    log_rank = np.log(np.array(range(1,101)))
    plt.plot(log_rank, log_freq)
    plt.xlabel('log rank')
    plt.ylabel('log freq')
    plt.title(i)
    plt.savefig(i + "_plot", bbox_inches='tight')
