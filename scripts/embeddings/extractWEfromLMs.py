import os
os.environ['TRANSFORMERS_CACHE'] = '/netscratch/cristinae/culture/cache/'

#from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertTokenizer, BertModel 
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
import numpy as np
import pandas as pd


bertML = 0
bertES = 0
bertDE = 0
bertIT = 0
bertEN = 0
bertAR = 0
bertTR = 1
xlm = 0
xglm = 0  # layer 47tb
# Parameters
layer = 11  # word embeddings    
fileName = 'bertEMBtr.layer'+str(layer)+'.vec'

# Load model
if(bertML):
  tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
  model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertEN):
  tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
  model = BertModel.from_pretrained('bert-base-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertDE):
  tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
  model = BertModel.from_pretrained('bert-base-german-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertES):
  tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
  model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertIT):
  tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
  model = BertModel.from_pretrained('dbmdz/bert-base-italian-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertTR):
  tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
  model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(bertAR):
  tokenizer = BertTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
  model = BertModel.from_pretrained('aubmindlab/bert-base-arabertv2',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 768
elif(xlm):
  tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
  model = XLMRobertaModel.from_pretrained("xlm-roberta-large",
                                output_hidden_states = True, # Whether the model returns all hidden-states.
                                )
  dim = 1024
elif(xglm):
  tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-2.9B')
  model = XGLMForCausalLM.from_pretrained('facebook/xglm-2.9B',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
  dim = 2048



# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Read input vocabulary
files = ['./data/weat_trads.tsv','./data/weat_origs.tsv']
#files = ['./data/kk.csv']
concepts = ['FRUITS','FLOWERS','INSECTS','INSTRUMENTS','WEAPONS','PLEASANT','UNPLEASANT']
#itemsWEAT = set() # we don't want duplicates
itemsWEAT = set(['España', 'Cataluña', 'Colombia', 'México', 'Bolivia', 'Ecuador', 'Chile', 'Argentina', 'Estados Unidos', 'Europa', 'América', 'Latinoamérica'])
itemsWEAT = set(['England', 'US', 'America', 'fruta', 'arma', 'instrumento', 'insecto', 'flor'])
for fileLists in files:
    for concept in concepts:
        df = pd.read_csv(fileLists, sep='\t',index_col=False)
        itemsWEAT.update(','.join(df[concept]).replace(', ', ',').split(','))

vecsWEAT = []
itemsWEATnb = [] # we will replace blanks with '_' in phrases
for text in itemsWEAT:
    # Add the special tokens.
    if(xlm or xglm):
        marked_text = "<s> " + text + " </s>"  #xml
    else:
        marked_text = "[CLS] " + text + " [SEP]" #bert
    #marked_text = text
    # Split the word into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12+1 layers.
    with torch.no_grad():
         outputs = model(tokens_tensor, segments_tensors)
    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
         # hidden_states[layer_i][batch_i][token_i])
         hidden_states = outputs[2]

    layer_i = layer
    batch_i = 0
    #vecsWEAT.append(hidden_states[layer_i][batch_i][token_i].numpy())
    tmp = np.zeros(dim)
    for i in range(1, len(tokenized_text)-1):
        print(tokenized_text[i])
        tmp = tmp + hidden_states[layer_i][batch_i][i].numpy()
    print('---')
    vecsWEAT.append(np.around(tmp, decimals=6))
    itemsWEATnb.append(text.rstrip().lstrip().replace(' ','_'))

vecsWEATw2vformat = [str(i) + ' '+ ' '.join(str(x) for x in j)  for i, j in zip(itemsWEATnb, vecsWEAT)]
with open(fileName, 'w') as f:
    for item in vecsWEATw2vformat:
        f.write("%s\n" % item)
