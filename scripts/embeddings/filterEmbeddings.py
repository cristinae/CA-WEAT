import pandas as pd

embeddingFile = 'cc100_17.tokLC.enes.vec' 
outputFile = embeddingFile+'.filtered'

# Read input vocabulary
files = ['../../data/weat_trads.tsv','../../data/weat_origs.tsv']
concepts = ['FLOWERS','INSECTS','INSTRUMENTS','WEAPONS','PLEASANT','UNPLEASANT']
itemsWEAT = set() # we don't want duplicates
for fileLists in files:
    for concept in concepts:
        df = pd.read_csv(fileLists, sep='\t',index_col=False)
        itemsWEAT.update(','.join(df[concept]).replace(', ', ',').lower().split(','))

# iterate over the embedding file and print the words in the list
with open(embeddingFile, "r") as fileIN, open(outputFile, "w") as fileOUT:
    for line in fileIN:
         word = line.split(' ', 1)[0]
         if word in itemsWEAT:
            fileOUT.write(line)
            

