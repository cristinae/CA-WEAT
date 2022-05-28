import re
import numpy as np
import pandas as pd
from collections import Counter

language = 'tr'
#prefixS = './'+language+'/sigmas'
prefixS = './rest/sigmas'
models = [
'wiki',
'wikiAlign',
'ccwp',
'cc10017',
'cc10017Align',
'cc10017vecmap',
'cc10017vecmapSup',
'w2v2langs',
'w2v9langs',
'bert0', 
'bert11',
#'bert0'+language, 
#'bert11'+language,
'xlmr0',
'xlmr11',
'xglm0',
'xglm47']
#models = ['cc10017']

def cleanListtoSet(rawList):
    clean = set()
    for elem in rawList:
        if elem.strip() != '':
           clean.add(elem.strip().lower()) # we lowercase because this is only used with word2vec    
    return clean

def loadCounts(filename):
    itemsCounter = Counter()
    with open(filename) as file:
        for line in file:
            word, freq = line.strip().split(' ') 
            dict.update(itemsCounter, {word: int(freq)})      
    return itemsCounter


print('Initialising...')
# The results will be here
df = pd.DataFrame()

# Loading the content of the word lists for the counts
fileTrads ='../data/weat_trads.tsv'
fileOrigs ='../data/weat_origs.tsv'
dfListTr = pd.read_csv(fileTrads, sep='\t',index_col=False)
dfListOr = pd.read_csv(fileOrigs, sep='\t',index_col=False)

# Loading the counts for CCe per language
countsCC = loadCounts('../text/counts/cc100_17.tokLC.'+language+'.counts')
if (language == 'en'):
   counts2langs = countsCC
else:
   counts2langs = countsCC + loadCounts('../text/counts/cc100_17.tokLC.en.counts')
counts9langs = Counter()
print('Loading counts...')
for languages in ['ar','ca','de','en','es','hr','it','tr','ru']:
   print(languages)
   counts9langs.update(loadCounts('../text/counts/cc100_17.tokLC.'+languages+'.counts')) 

# Loading isomorphism measures
isodf = pd.read_csv('../data/isomorfisme.csv') 
    
init = 0
for test in range(1,3):
  print("WEAT"+str(test))
  for model in models:

    statistic = []
    sizeEffect = []
    statisticSig = []
    statisticSigUp = []
    statisticSigLo = []
    sizeEffectSig = []   
    sizeEffectSigUp = []   
    sizeEffectSigLo = []
    definition = []   
    labels = []
    label = 1
    countsFruits = []
    countsFlowers = []
    countsInsects = []
    countsInstruments = []
    countsWeapons = []
    countsPleasant = []
    countsUnpleasant = []
    for i in range(1,30):
    #for i in ["_ES1", "_ES2", "_ES3", "_EC1", "_EC2", "_BO1", "_CO1", "_CO2", "_MX1", "_MX2"]:
    #for i in ["_US1", "_US2", "_US3", "_US4", "_US5"]:
        # size effects and statistics
        instance = language +str(i)
        fileNameSig = prefixS+'/ca_'+model+'_'+ instance +'_cosine_'+str(test)+'_uncased.res'
        res = []
        means = []
        sigmas = []
        # we read the results with boostrapped lists
        try:
          with open (fileNameSig, 'rt') as resultsFile: 
             contents2 = resultsFile.read() 
             res = re.findall(r'-*[0|1|2]\.\d\d\d\d\d\d\d\d\d', contents2)
             means = re.findall(r'(-*\d\.\d\d)\$', contents2)
             sigmas = re.findall(r'([0|1|2]\.\d\d)\}', contents2)
        except FileNotFoundError:
           print("skipping " + instance)
           continue
        statistic.append(float(res[0]))
        sizeEffect.append(float(res[3]))
        statisticSig.append(float(means[0])) 
        statisticSigUp.append(float(sigmas[0]))
        statisticSigLo.append(float(sigmas[1])) 
        sizeEffectSig.append(float(means[1]))
        sizeEffectSigUp.append(float(sigmas[2]))
        sizeEffectSigLo.append(float(sigmas[3]))
        labels.append(label)
        label = label+1
        definition.append(instance)
        # counts 
        if(model.startswith('cc10017') or model.startswith('w2v')):
           fruits = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['FRUITS'].values[0].replace(', ', ',').split(','))
           flowers = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['FLOWERS'].values[0].replace(', ', ',').split(','))
           insects = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['INSECTS'].values[0].replace(', ', ',').split(','))
           instruments = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['INSTRUMENTS'].values[0].replace(', ', ',').split(','))
           weapons = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['WEAPONS'].values[0].replace(', ', ',').split(','))
           pleasant = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['PLEASANT'].values[0].replace(', ', ',').split(','))
           unpleasant = cleanListtoSet(dfListOr.loc[dfListOr['LANG']==instance]['UNPLEASANT'].values[0].replace(', ', ',').split(','))
           cFruits = 0
           cFlowers = 0
           cInsects = 0
           cInstruments = 0
           cWeapons = 0
           cPleasant = 0
           cUnpleasant = 0
        if(model.startswith('cc10017')):
           for elem in fruits:
               cFruits = cFruits+countsCC[elem]
           for elem in flowers:
               cFlowers = cFlowers+countsCC[elem]
           for elem in insects:
               cInsects = cInsects+countsCC[elem]
           for elem in instruments:
               cInstruments = cInstruments+countsCC[elem]
           for elem in weapons:
               cWeapons = cWeapons+countsCC[elem]
           for elem in pleasant:
               cPleasant = cPleasant+countsCC[elem]
           for elem in unpleasant:
               cUnpleasant = cUnpleasant+countsCC[elem]
        if(model == 'w2v2langs'):
           for elem in fruits:
               cFruits = cFruits+counts2langs[elem]
           for elem in flowers:
               cFlowers = cFlowers+counts2langs[elem]
           for elem in insects:
               cInsects = cInsects+counts2langs[elem]
           for elem in instruments:
               cInstruments = cInstruments+counts2langs[elem]
           for elem in weapons:
               cWeapons = cWeapons+counts2langs[elem]
           for elem in pleasant:
               cPleasant = cPleasant+counts2langs[elem]
           for elem in unpleasant:
               cUnpleasant = cUnpleasant+counts2langs[elem]
        if(model == 'w2v9langs'):
           for elem in fruits:
               cFruits = cFruits+counts9langs[elem]
           for elem in flowers:
               cFlowers = cFlowers+counts9langs[elem]
           for elem in insects:
               cInsects = cInsects+counts9langs[elem]
           for elem in instruments:
               cInstruments = cInstruments+counts9langs[elem]
           for elem in weapons:
               cWeapons = cWeapons+counts9langs[elem]
           for elem in pleasant:
               cPleasant = cPleasant+counts9langs[elem]
           for elem in unpleasant:
               cUnpleasant = cUnpleasant+counts9langs[elem]
        if(model.startswith('cc10017') or model.startswith('w2v')):
           countsFruits.append(int(cFruits))
           countsFlowers.append(int(cFlowers))
           countsInsects.append(int(cInsects))
           countsInstruments.append(int(cInstruments))
           countsWeapons.append(int(cWeapons))
           countsPleasant.append(int(cPleasant))
           countsUnpleasant.append(int(cUnpleasant))

        
    # we read the results with the translated boostrapped list
    fileNameSigTrad = prefixS+'/trads/w2v_'+model+'_'+language+'_cosine_'+str(test)+'_uncased.res'
    with open (fileNameSigTrad, 'rt') as resultsFile: 
        contents = resultsFile.read() 
    meansTrad = re.findall(r'(-*\d\.\d\d)\$', contents)
    sigmasTrad = re.findall(r'([0|1|2]\.\d\d)\}', contents)

    # Statistic medians
    median = np.percentile(statisticSig,50)
    ci_high = np.percentile(statisticSig,95)
    ci_low = np.percentile(statisticSig,5)
    max_ci = '{:.2f}'.format(round(ci_high-median,2))
    min_ci = '{:.2f}'.format(round(median-ci_low,2))
    medianRes = '$'+'{:.2f}'.format(round(median,2))+ '^{+' + max_ci + '}_{-' + min_ci +'}$'
    transRes = '$'+meansTrad[0]+ '^{+' + sigmasTrad[0] + '}_{-' + sigmasTrad[1] +'}$'
    
    # Size effect medians
    medianE = np.percentile(sizeEffect,50)
    ci_highE = np.percentile(sizeEffect,95)
    ci_lowE = np.percentile(sizeEffect,5)
    max_ciE = '{:.2f}'.format(round(ci_highE-medianE,2))
    min_ciE = '{:.2f}'.format(round(medianE-ci_lowE,2))
    medianResE = '$'+'{:.2f}'.format(round(medianE,2))+ '^{+' + max_ciE + '}_{-' + min_ciE +'}$'
    transResE = '$'+meansTrad[1]+ '^{+' + sigmasTrad[2] + '}_{-' + sigmasTrad[3] +'}$'

    # Counts
    if(not model.startswith('cc10017') and not model.startswith('w2v')):    
       countsFruits = '-'
       countsFlowers = '-'
       countsInsects = '-'
       countsInstruments = '-'
       countsWeapons = '-'
       countsPleasant = '-'
       countsUnpleasant = '-'
       
    # Isometry
    if model in isodf.values and language != 'en':
       EV = isodf.loc[isodf['model'] == model][language+'EV'].values[0]
       GH = isodf.loc[isodf['model'] == model][language+'GH'].values[0]
    else:
       EV = '-'
       GH = '-'
       
    if (init==1):
       tmp = pd.DataFrame()
       tmp.insert(0, 'language', definition)
       tmp.insert(1, 'source', 1)
       tmp.insert(2, 'test', str(test))
       tmp.insert(3, 'model', model)
       tmp.insert(4, 'statistic', statistic)
       tmp.insert(5, 'DstatUP', statisticSigUp)
       tmp.insert(6, 'DstatLO', statisticSigLo)
       tmp.insert(7, 'statMedian', median)
       tmp.insert(8, 'DstatMedUP', max_ci)
       tmp.insert(9, 'DstatMedLO', min_ci)
       tmp.insert(10, 'sizeEffect', sizeEffect)
       tmp.insert(11, 'DsizeEffectUP', sizeEffectSigUp)
       tmp.insert(12, 'DsizeEffectLO', sizeEffectSigLo)
       tmp.insert(13, 'effMedian', medianE)
       tmp.insert(14, 'DeffMedUP', max_ciE)
       tmp.insert(15, 'DeffMedLO', min_ciE)
       tmp.insert(16, 'cFruits', countsFruits)
       tmp.insert(17, 'cFlowers', countsFlowers)
       tmp.insert(18, 'cInsects', countsInsects)
       tmp.insert(19, 'cInstruments', countsInstruments)
       tmp.insert(20, 'cWeapons', countsWeapons)
       tmp.insert(21, 'cPleasant', countsPleasant)
       tmp.insert(22, 'cUnpleasant', countsUnpleasant)
       tmp.insert(23, 'EV', EV)
       tmp.insert(24, 'GH', GH)
       df=df.append(tmp)
    else:
       df.insert(0, 'language', definition)
       df.insert(1, 'source', 1)
       df.insert(2, 'test', str(test))
       df.insert(3, 'model', model)
       df.insert(4, 'statistic', statistic)
       df.insert(5, 'DstatUP', statisticSigUp)
       df.insert(6, 'DstatLO', statisticSigLo)
       df.insert(7, 'statMedian', median)
       df.insert(8, 'DstatMedUP', max_ci)
       df.insert(9, 'DstatMedLO', min_ci)
       df.insert(10, 'sizeEffect', sizeEffect)
       df.insert(11, 'DsizeEffectUP', sizeEffectSigUp)
       df.insert(12, 'DsizeEffectLO', sizeEffectSigLo)
       df.insert(13, 'effMedian', medianE)
       df.insert(14, 'DeffMedUP', max_ciE)
       df.insert(15, 'DeffMedLO', min_ciE)
       df.insert(16, 'cFruits', countsFruits)
       df.insert(17, 'cFlowers', countsFlowers)
       df.insert(18, 'cInsects', countsInsects)
       df.insert(19, 'cInstruments', countsInstruments)
       df.insert(20, 'cWeapons', countsWeapons)
       df.insert(21, 'cPleasant', countsPleasant)
       df.insert(22, 'cUnpleasant', countsUnpleasant)
       df.insert(23, 'EV', EV)
       df.insert(24, 'GH', GH)
       init=1
    
    # X_WEAT
    # counts 
    if(model.startswith('cc10017') or model.startswith('w2v')):
       #fruits = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==instance]['FRUITS'].values[0].replace(', ', ',').split(','))
       flowers = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['FLOWERS'].values[0].replace(', ', ',').split(','))
       insects = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['INSECTS'].values[0].replace(', ', ',').split(','))
       instruments = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['INSTRUMENTS'].values[0].replace(', ', ',').split(','))
       weapons = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['WEAPONS'].values[0].replace(', ', ',').split(','))
       pleasant = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['PLEASANT'].values[0].replace(', ', ',').split(','))
       unpleasant = cleanListtoSet(dfListTr.loc[dfListTr['LANG']==language]['UNPLEASANT'].values[0].replace(', ', ',').split(','))
       #cFruits = 0
       cFlowers = 0
       cInsects = 0
       cInstruments = 0
       cWeapons = 0
       cPleasant = 0
       cUnpleasant = 0
    if(model.startswith('cc10017')):
       #for elem in fruits:
       #    cFruits = cFruits+countsCC[elem]
       for elem in flowers:
           cFlowers = cFlowers+countsCC[elem]
       for elem in insects:
           cInsects = cInsects+countsCC[elem]
       for elem in instruments:
           cInstruments = cInstruments+countsCC[elem]
       for elem in weapons:
           cWeapons = cWeapons+countsCC[elem]
       for elem in pleasant:
           cPleasant = cPleasant+countsCC[elem]
       for elem in unpleasant:
           cUnpleasant = cUnpleasant+countsCC[elem]
    if(model == 'w2v2langs'):
       #for elem in fruits:
       #    cFruits = cFruits+counts2langs[elem]
       for elem in flowers:
           cFlowers = cFlowers+counts2langs[elem]
       for elem in insects:
           cInsects = cInsects+counts2langs[elem]
       for elem in instruments:
           cInstruments = cInstruments+counts2langs[elem]
       for elem in weapons:
           cWeapons = cWeapons+counts2langs[elem]
       for elem in pleasant:
           cPleasant = cPleasant+counts2langs[elem]
       for elem in unpleasant:
           cUnpleasant = cUnpleasant+counts2langs[elem]
    if(model == 'w2v9langs'):
       #for elem in fruits:
       #    cFruits = cFruits+counts9langs[elem]
       for elem in flowers:
           cFlowers = cFlowers+counts9langs[elem]
       for elem in insects:
           cInsects = cInsects+counts9langs[elem]
       for elem in instruments:
           cInstruments = cInstruments+counts9langs[elem]
       for elem in weapons:
           cWeapons = cWeapons+counts9langs[elem]
       for elem in pleasant:
           cPleasant = cPleasant+counts9langs[elem]
       for elem in unpleasant:
           cUnpleasant = cUnpleasant+counts9langs[elem]
    if(not model.startswith('cc10017') and not model.startswith('w2v')):    
       #cFruits = '-'
       cFlowers = '-'
       cInsects = '-'
       cInstruments = '-'
       cWeapons = '-'
       cPleasant = '-'
       cUnpleasant = '-'    
    # Adding a row for translations. Notice that CI always come from bootstraping and the medians are just a copy of the value to complete the table
    traslationRow = {'language':language+'0', 'source':0,  'test':str(test), 'model':model, 'statistic':meansTrad[0], 'DstatUP':sigmasTrad[0], 'DstatLO':sigmasTrad[1], 'statMedian':meansTrad[0], 'DstatMedUP':sigmasTrad[0], 'DstatMedLO':sigmasTrad[1], 'sizeEffect':meansTrad[1],  'DsizeEffectUP':sigmasTrad[2], 'DsizeEffectLO':sigmasTrad[3], 'effMedian':meansTrad[1],  'DeffMedUP':sigmasTrad[2], 'DeffMedLO':sigmasTrad[3], 'cFruits':'-', 'cFlowers':cFlowers, 'cInsects':cInsects, 'cInstruments':cInstruments, 'cWeapons':cWeapons, 'cPleasant':cPleasant,'cUnpleasant':cUnpleasant,'EV':EV, 'GH':GH}
    df = df.append(traslationRow, ignore_index=True)

    print(df)
    df.to_csv('collectedData.'+language+'.csv', index=False)
    #print(df.to_csv(index=False))


