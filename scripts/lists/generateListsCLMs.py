import os
os.environ['TRANSFORMERS_CACHE'] = '/netscratch/cristinae/culture/cache/'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import XGLMTokenizer, XGLMForCausalLM

import numpy as np

toker = XGLMTokenizer.from_pretrained('facebook/xglm-2.9B')
model = XGLMForCausalLM.from_pretrained('facebook/xglm-2.9B')
#toker = AutoTokenizer.from_pretrained("gpt2-large")
#model = AutoModelForCausalLM.from_pretrained("gpt2-large")
#toker = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
#model = AutoModelForCausalLM.from_pretrained("bert-base-multilingual-cased", is_decoder=True)


print('starting')
# Read sentences to complete
with open('./data/querySentences.tsv') as f:
    sentences = f.readlines()

maxWords = 4
iniWords = 0
numPhrases = 25
for sentence in sentences:
    print(sentence)
    columns = sentence.split('\t')
    # xgml_25Fruit_1-2.en
    fileNameOut = './genLists/xgml_'+str(numPhrases)+columns[0]+'_'+columns[1]+'.'+columns[2]
    fileNameOut = fileNameOut.replace(':','-')
    seqIni = columns[3]
    #seqIni = "My favourite fruit is the"
    #print("\nInput sequence: ")
    print(fileNameOut)
    print('"'+seqIni+'"')

    inpts = toker(seqIni, return_tensors="pt")
    #print("\nTokenized input data structure: ")
    #print(inpts)

    with torch.no_grad():
        logits = model(**inpts).logits[:, -1, :]
    #print("\nAll logits for next word: ")
    #print(logits)
    #print(logits.shape)

    #pred_id = torch.argmax(logits).item()
    #print("\nPredicted token ID of next word: ")
    #print(toker.convert_ids_to_tokens(pred_id))

    pred_topid = torch.topk(logits, numPhrases)
    sequences = []
    phrase = ''
    seq = ''
    for elemID in pred_topid[1][0]:
        elem = toker.convert_ids_to_tokens(elemID.item())
        if (elem[0]=='▁'):
            iniWords = iniWords+1
            seq = seqIni + toker.decode(elemID)
            phrase = toker.decode(elemID)
        while (iniWords<maxWords):
            #print(seq)
            newInpts = toker(seq, return_tensors="pt")
            with torch.no_grad():
               newLogits = model(**newInpts).logits[:, -1, :]
            nextTok = toker.convert_ids_to_tokens(torch.argmax(newLogits).item())
            #print(nextTok)
            if (nextTok[0]=='▁'):
                iniWords = iniWords+1
            if (nextTok=='.' or nextTok==','):
                break
            elif (nextTok==toker.eos_token):
                break
            seq = seq + nextTok.replace('▁',' ')
            phrase = phrase + nextTok.replace('▁',' ')
            #print(phrase)
        iniWords = 0  
        sequences.append(phrase)
    #print(seq)

    with open(fileNameOut, 'w') as f:
        for seq in sequences:
            f.write("%s\n" % seq)

print("\nEnd")

