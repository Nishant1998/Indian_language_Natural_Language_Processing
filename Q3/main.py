
import numpy as np
import pandas as pd
import pickle as pk
import copy
import re
from collections import Counter
import os, psutil  

endv = np.array(['़','ा','ि','ी','ु','ू','ृ','ॅ','ॆ','े','ै','ॉ','ो','ौ','्', 'ँ','ं','ः'])

def getCharList(line):
    # split in char
    line = u" ".join(line)
    line = line.split() # list of line char
    # correct hallant
    for i in range(len(line)-1):
        if line[i+1] not in endv and line[i] not in endv:
            line[i] = line[i]+ '्अ'
        
    # make char list
    line = [u' '.join(i) for i in line]
    string = ""
    for i in line:
        string = string  + i
    string = u" ".join(string)
    string = string.split()
    return string

def get_syllable(List):
    char = ""
    syllable_list = [] 
    for i in List:
        char = char + i
        if i in endv:
            syllable_list.append(char)
            char = ""
    return syllable_list

def clean_line(line):
    regex= re.compile('[A-Za-z0-9०-९]*[$&+,:;=?@#|\'<>.-^*()%!]*')
    line  = re.sub(regex, "", line)
    return line

def get_unigram(List):
    return Counter(List)

def get_bigram(line, mode = 0):
    bigram_list = []
    for i in range(len(line) - 1):
        if mode == 0:
            char = line[i] + line[i+1]
        elif mode == 1:
            char = line[i] + " " + line[i+1]
        bigram_list.append(char)
    return Counter(bigram_list)

def get_trigram(line, mode = 0):
    trigram_list = []
    for i in range(len(line) - 2):
        if mode == 0:
            char = line[i] + line[i+1] + line[i+2]
        elif mode == 1:
            char = line[i] + " " + line[i+1] + " " + line[i+2]
        
        trigram_list.append(char)
    return Counter(trigram_list)

def get_quardrigram(line):
    quardrigram_list = []
    for i in range(len(line) - 3):
        char = line[i] + line[i+1] + line[i+2] + line[i+3]
        quardrigram_list.append(char)
    return Counter(quardrigram_list)

def save(name, data):
    df = pd.DataFrame(columns=[name,'frequency'])
    for i in data:
        df.loc[len(df)] = list(i)
    df.to_csv(str(name) + ".csv",index=False)
    print("Saved : " + name)
    
def check_memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return np.round(memory_use, 2)

def save_top100():
    print("combine results ")
    file_name = [i for i in os.listdir() if len(i) <3]
    print(file_name)
    f1 = Counter()
    f2 = Counter()
    f3 = Counter()
    f4 = Counter()
    f5 = Counter()
    f6 = Counter()
    f7 = Counter()
    f8 = Counter()
    f9 = Counter()
    f10 = Counter()
    
    for i in file_name:
        with open(i,'rb') as file:
            [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10] = pk.load(file)
            f1.update(Counter(dict(d1)))
            f2.update(Counter(dict(d2)))
            f3.update(Counter(dict(d3)))
            f4.update(Counter(dict(d4)))
            f5.update(Counter(dict(d5)))
            f6.update(Counter(dict(d6)))
            f7.update(Counter(dict(d7)))
            f8.update(Counter(dict(d8)))
            f9.update(Counter(dict(d9)))
            f10.update(Counter(dict(d10)))
    
    print("Saving Files.")
    
    unigram_count = f1.most_common(100)
    save('unigram_char' , unigram_count)
    # 2
    bigram_count = f2.most_common(100)
    save('bigram_char' , bigram_count)
    # 3
    trigram_count = f3.most_common(100)
    save('trigram_char' , trigram_count)
    # 4
    quardrigram_count = f4.most_common(100)
    save('quardrigram_char' , quardrigram_count)
    # save for word
    # 1
    unigram_word = f5.most_common(100)
    save('unigram_word' , unigram_word)
    # 2
    bigram_word = f6.most_common(100)
    save('bigram_word' , bigram_word)
    # 3
    trigram_word = f7.most_common(100)
    save('trigram_word' , trigram_word)
    # save for syllable
    # 1
    unigram_syllable = f8.most_common(100)
    save('unigram_syllable' , unigram_syllable)
    # 2
    bigram_syllable = f9.most_common(100)
    save('bigram_syllable' , bigram_syllable)
    # 3
    trigram_syllable = f10.most_common(100)
    save('trigram_syllable' , trigram_syllable)


def save_count(List, ittr):
    with open(str(ittr), 'wb') as file:
        pk.dump(List, file)

def main(): 
    path =  "mr.txt"
    
    unigram_count = Counter()
    bigram_count = Counter()
    trigram_count = Counter()
    quardrigram_count = Counter()
    
    unigram_word = Counter()
    bigram_word = Counter()
    trigram_word = Counter()
    
    unigram_syllable = Counter()
    bigram_syllable = Counter()
    trigram_syllable = Counter()
    
    ittr = 0
    # read data # 1gb 1073741824 536870912 268435456
    with open(path,"r") as file:
        for num, line in enumerate(file, 1):
            print(str(num) + " / " + '33976000  :  ' + str(int((num/33976000)*100)) + " %  : " +  str(check_memory()) + " GB")
            line  = clean_line(line)
            charList = getCharList(line)
            wordList = line.split()
            
            # A
            unigram_count.update(get_unigram(charList))
            bigram_count.update(get_bigram(charList))
            trigram_count.update(get_trigram(charList))
            quardrigram_count.update(get_quardrigram(charList))
    
            # B
            unigram_word.update(get_unigram(wordList))
            bigram_word.update(get_bigram(wordList,1))
            trigram_word.update(get_trigram(wordList,1))
    
            # C
            syllable_list = get_syllable(charList)
            unigram_syllable.update(get_unigram(syllable_list))
            bigram_syllable.update(get_bigram(syllable_list))
            trigram_syllable.update(get_trigram(syllable_list))
            
            #if num == 16988000:
            #    pass
            if num % 849400 == 0:
                print("reset..")
                print(check_memory())
                d1 = unigram_count.most_common(100)
                d2 = bigram_count.most_common(100)
                d3 = trigram_count.most_common(100)
                d4 = quardrigram_count.most_common(100)
                d5 = unigram_word.most_common(100)
                d6 = bigram_word.most_common(100)
                d7 = trigram_word.most_common(100)
                d8 = unigram_syllable.most_common(100)
                d9 = bigram_syllable.most_common(100)
                d10 = trigram_syllable.most_common(100)
                save_count([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10], ittr)
                ittr = ittr + 1
                
                unigram_count = Counter()
                bigram_count = Counter()
                trigram_count = Counter()
                quardrigram_count = Counter()
                unigram_word = Counter()
                bigram_word = Counter()
                trigram_word = Counter()
                unigram_syllable = Counter()
                bigram_syllable = Counter()
                trigram_syllable = Counter()


    save_top100()
    
if __name__ == "__main__":
    save_top100()
