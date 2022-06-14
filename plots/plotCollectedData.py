import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import re
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly

dataFile = 'collectedData.csv'
language = 'XX'

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
'bert0'+language, 
'bert11'+language,
'bert0', 
'bert11',
'xlmr0',
'xlmr11',
'xglm0',
'xglm47']
#models = ['wiki']

naming = {
'wiki':'WP',
'wikiAlign':'WPali',
'ccwp':'CCWP',
'cc10017':'CCe',
'cc10017Align':'CCeAli',
'cc10017vecmap':'CCeVMuns',
'cc10017vecmapSup':'CCeVMsup',
'w2v2langs':'CCe2langs',
'w2v9langs':'CCe9langs',
'bert0':'mBERT$_{0}$', 
'bert11':'mBERT$_{11}$',
'bert0'+language:'BERT$_{0}$', 
'bert11'+language:'BERT$_{11}$',
'bert0es':'BERT$_{0}$', 
'bert11es':'BERT$_{11}$',
'bert0de':'BERT$_{0}$', 
'bert11de':'BERT$_{11}$',
'bert0it':'BERT$_{0}$', 
'bert11it':'BERT$_{11}$',
'bert0en':'BERT$_{0}$', 
'bert11en':'BERT$_{11}$',
'xlmr0':'XLM-R$_{0}$',
'xlmr11':'XLM-R$_{11}$',
'xglm0':'XGLM$_{0}$',
'xglm47':'XGLM$_{47}$'
}


plt.rcParams.update({
 "axes.linewidth":2,
 "xtick.major.width":2,
 "xtick.minor.width":2,
 "ytick.major.width":2,
 "ytick.minor.width":2,
 "xtick.major.size":8,
 "ytick.major.size":8,
 "xtick.minor.size":6,
 "ytick.minor.size":6
})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 20
})
fontTit = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'heavy',
        'size': 24,
        }
fontAx = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 24,
        }


def plotCountsXweat(df, test):

   model = 'cc10017'
#   model = 'cc10017vecmap'
   variable = 'sizeEffect'
   #variable = 'effMedian'
#   model = 'w2v9langs'
   positive = 'cPleasant'
   negative = 'cUnpleasant'
#   positive = 'cFlowers'
#   negative = 'cInsects'
   dfWithCounts = df[(~df[positive].str.contains('-')) & (df['test']==test) & (df['model'].str.fullmatch(model))  & (df['source']==0)  ]
   #dfWithCounts = df[(~df[positive].str.contains('-')) & (df['test']==test) & (df['model'].str.startswith(model))  & (df['source']==0)  ]
   countsAtt=dfWithCounts[positive].astype('int')-dfWithCounts[negative].astype('int')
   asymmetric_error = np.array(list(zip(dfWithCounts['D'+variable+'LO'], dfWithCounts['D'+variable+'UP']))).T

   fig = plt.figure(figsize=(9.5, 5))
   plt.errorbar(countsAtt, dfWithCounts[variable], yerr=asymmetric_error, fmt='none', ecolor = 'darkred', color='lightslategrey')

   dfHR = dfWithCounts[dfWithCounts['language'].str.startswith('hr')]
   countsAttHR=dfHR[positive].astype('int')-dfHR[negative].astype('int')
   plt.plot(countsAttHR.mean(), dfHR[variable].values[0], 'yX', markersize=11)

   dfIT = dfWithCounts[dfWithCounts['language'].str.startswith('it')]
   countsAttIT=dfIT[positive].astype('int')-dfIT[negative].astype('int')
   plt.plot(countsAttIT.mean(), dfIT[variable].values[0], 'gs', markersize=11)

   dfES = dfWithCounts[dfWithCounts['language'].str.startswith('es')]
   countsAttES=dfES[positive].astype('int')-dfES[negative].astype('int')
   plt.plot(countsAttES.mean(), dfES[variable].values[0], 'bp', markersize=11)

   dfDE = dfWithCounts[dfWithCounts['language'].str.startswith('de')]
   countsAttDE=dfDE[positive].astype('int')-dfDE[negative].astype('int')
   plt.plot(countsAttDE.mean(), dfDE[variable].values[0], 'rP', markersize=11)

   dfAR = dfWithCounts[dfWithCounts['language'].str.startswith('ar')]
   countsAttAR=dfAR[positive].astype('int')-dfAR[negative].astype('int')
   plt.plot(countsAttAR.mean(), dfAR[variable].values[0], 'm<', markersize=11)

   dfCA = dfWithCounts[dfWithCounts['language'].str.startswith('ca')]
   countsAttCA=dfCA[positive].astype('int')-dfCA[negative].astype('int')
   plt.plot(countsAttCA.mean(), dfCA[variable].values[0], 'mv', markersize=11)

   dfRU = dfWithCounts[dfWithCounts['language'].str.startswith('ru')]
   countsAttRU=dfRU[positive].astype('int')-dfRU[negative].astype('int')
   plt.plot(countsAttRU.mean(), dfRU[variable].values[0], 'm^', markersize=11)

   dfTR = dfWithCounts[dfWithCounts['language'].str.startswith('tr')]
   countsAttTR=dfTR[positive].astype('int')-dfTR[negative].astype('int')
   plt.plot(countsAttTR.mean(), dfTR[variable].values[0], 'm>', markersize=11)

   dfEN = dfWithCounts[dfWithCounts['language'].str.startswith('en')]
   countsAttEN=dfEN[positive].astype('int')-dfEN[negative].astype('int')
   plt.plot(countsAttEN.mean(), dfEN[variable].values[0], 'ko', markersize=11)

   legend_elements = [
                   Line2D([0], [0], color='g', marker='s', label='it', markersize=17),
                   Line2D([0], [0], color='r', marker='P', label='de', markersize=17),
                   Line2D([0], [0], color='y', marker='X', label='hr', markersize=17),
                   Line2D([0], [0], color='b', marker='p', label='es', markersize=17),
                   Line2D([0], [0], color='m', marker='<', label='ar', markersize=17),
                   Line2D([0], [0], color='m', marker='v', label='ca', markersize=17),
                   Line2D([0], [0], color='m', marker='^', label='ru', markersize=17),
                   Line2D([0], [0], color='m', marker='>', label='tr', markersize=17)
                   ,Line2D([0], [0], color='k', marker='o', label='en', markersize=17)
                   ]

   mb, stats = poly.polyfit(countsAtt, dfWithCounts[variable], 1, full=True)
   SSE = stats[0][0]
   diff2 = (dfWithCounts[variable] - dfWithCounts[variable].mean())**2
   SST = diff2.sum()
   r2 = 1-SSE/SST 
   print('r2=',r2)
   plt.plot(np.array(countsAtt), mb[1]*np.array(countsAtt) + mb[0])

   plt.ylabel('Effect size $d$', fontdict=fontAx)
   #plt.xlabel('$\Delta$Counts (Pleasant-Unpleasant)', fontdict=fontAx)
   plt.xlabel(positive + ' - ' + negative, fontdict=fontAx)
   plt.axis([-2000000, 5500000, -2, 2])
   plt.yticks(np.arange(-2, 2.01, 1.0))
#   plt.title('    '+naming[model], fontdict=fontTit)
   plt.tight_layout(pad=0.03)
   plt.legend(handles=legend_elements, ncol=5, loc='lower center', frameon=False)
   #plt.show()
   plt.savefig('freqsXWEAT'+naming[model]+'WEAT'+str(test)+'.png')
   plt.clf()
    
    

def plotCounts(df, test):

   model = 'cc10017'
#   model = 'cc10017vecmap'
   variable = 'sizeEffect'
#   model = 'w2v9langs'
   positive = 'cPleasant'
   negative = 'cUnpleasant'
#   positive = 'cFlowers'
#   negative = 'cInsects'
   dfWithCounts = df[(~df[positive].str.contains('-')) & (df['test']==test) & (df['model'].str.fullmatch(model)) ]
   #dfWithCounts = df[(~df[positive].str.contains('-')) & (df['test']==test) & (df['model'].str.startswith(model))  & (df['source']==0)  ]
   countsAtt=dfWithCounts[positive].astype('int')-dfWithCounts[negative].astype('int')
   asymmetric_error = np.array(list(zip(dfWithCounts['D'+variable+'LO'], dfWithCounts['D'+variable+'UP']))).T

   fig = plt.figure(figsize=(9.5, 5))
   plt.errorbar(countsAtt, dfWithCounts[variable], yerr=asymmetric_error, fmt='none', ecolor = 'darkred', color='lightslategrey')

   dfHR = dfWithCounts[dfWithCounts['language'].str.startswith('hr')]
   countsAttHR=dfHR[positive].astype('int')-dfHR[negative].astype('int')
   plt.plot(countsAttHR, dfHR[variable], 'yX', markersize=11)

   dfIT = dfWithCounts[dfWithCounts['language'].str.startswith('it')]
   countsAttIT=dfIT[positive].astype('int')-dfIT[negative].astype('int')
   plt.plot(countsAttIT, dfIT[variable], 'gs', markersize=11)

   dfES = dfWithCounts[dfWithCounts['language'].str.startswith('es')]
   countsAttES=dfES[positive].astype('int')-dfES[negative].astype('int')
   plt.plot(countsAttES, dfES[variable], 'bp', markersize=11)

   dfDE = dfWithCounts[dfWithCounts['language'].str.startswith('de')]
   countsAttDE=dfDE[positive].astype('int')-dfDE[negative].astype('int')
   plt.plot(countsAttDE, dfDE[variable], 'rP', markersize=11)

   dfAR = dfWithCounts[dfWithCounts['language'].str.startswith('ar')]
   countsAttAR=dfAR[positive].astype('int')-dfAR[negative].astype('int')
   plt.plot(countsAttAR, dfAR[variable], 'm<', markersize=11)

   dfCA = dfWithCounts[dfWithCounts['language'].str.startswith('ca')]
   countsAttCA=dfCA[positive].astype('int')-dfCA[negative].astype('int')
   plt.plot(countsAttCA, dfCA[variable], 'mv', markersize=11)

   dfRU = dfWithCounts[dfWithCounts['language'].str.startswith('ru')]
   countsAttRU=dfRU[positive].astype('int')-dfRU[negative].astype('int')
   plt.plot(countsAttRU, dfRU[variable], 'm^', markersize=11)

   dfTR = dfWithCounts[dfWithCounts['language'].str.startswith('tr')]
   countsAttTR=dfTR[positive].astype('int')-dfTR[negative].astype('int')
   plt.plot(countsAttTR, dfTR[variable], 'm>', markersize=11)

   dfEN = dfWithCounts[dfWithCounts['language'].str.startswith('en')]
   countsAttEN=dfEN[positive].astype('int')-dfEN[negative].astype('int')
   plt.plot(countsAttEN, dfEN[variable], 'ko', markersize=11)

   legend_elements = [
                   Line2D([0], [0], color='g', marker='s', label='it', markersize=17),
                   Line2D([0], [0], color='r', marker='P', label='de', markersize=17),
                   Line2D([0], [0], color='y', marker='X', label='hr', markersize=17),
                   Line2D([0], [0], color='b', marker='p', label='es', markersize=17),
                   Line2D([0], [0], color='m', marker='<', label='ar', markersize=17),
                   Line2D([0], [0], color='m', marker='v', label='ca', markersize=17),
                   Line2D([0], [0], color='m', marker='^', label='ru', markersize=17),
                   Line2D([0], [0], color='m', marker='>', label='tr', markersize=17)
                   ,Line2D([0], [0], color='k', marker='o', label='en', markersize=17)
                   ]

   mb, stats = poly.polyfit(countsAtt, dfWithCounts[variable], 1, full=True)
   SSE = stats[0][0]
   diff2 = (dfWithCounts[variable] - dfWithCounts[variable].mean())**2
   SST = diff2.sum()
   r2 = 1-SSE/SST 
   print('r2=',r2)
   plt.plot(np.array(countsAtt), mb[1]*np.array(countsAtt) + mb[0])

   plt.ylabel('Effect size $d$', fontdict=fontAx)
   #plt.xlabel('$\Delta$Counts (Pleasant-Unpleasant)', fontdict=fontAx)
   plt.xlabel(positive + ' - ' + negative, fontdict=fontAx)
   plt.axis([-2000000, 5500000, -2, 2])
   plt.yticks(np.arange(-2, 2.01, 1.0))
   #plt.title('    '+naming[model], fontdict=fontTit)
   plt.tight_layout(pad=0.03)
   plt.legend(handles=legend_elements, ncol=5, loc='lower center', frameon=False)
   #plt.show()
   plt.savefig('freqsCAWEAT'+naming[model]+'WEAT'+str(test)+'.png')
   plt.clf()
    
    

def plotSExModel(df, test, weat):

   dfTrads = df[(df['source']==0) & (~df['model'].str.fullmatch('cc10017Align')) & (df['test']==test)  ]
   dfOrigs = df[(df['language'].str.endswith("1")) &  (~df['model'].str.fullmatch('cc10017Align')) & (df['test']==test)  ]
   asymmetric_errorTr = np.array(list(zip(dfTrads['DsizeEffectLO'], dfTrads['DsizeEffectUP']))).T
   asymmetric_errorOr = np.array(list(zip(dfOrigs['DeffMedLO'], dfOrigs['DeffMedUP']))).T

   dfTrads['labels'] = dfTrads.apply(lambda L: L.model.replace(L.model, naming[L.model]), axis=1)
   dfOrigs['labels'] = dfOrigs.apply(lambda L: L.model.replace(L.model, naming[L.model]), axis=1)
   fig = plt.figure(figsize=(9.5, 6))
   if (weat=='tr'):
     plt.errorbar(dfTrads['labels'], dfTrads['sizeEffect'], yerr=asymmetric_errorTr, fmt='none', ecolor = 'darkred', color='lightslategrey')
   else:
     plt.errorbar(dfOrigs['labels'], dfOrigs['effMedian'], yerr=asymmetric_errorOr, fmt='none', ecolor = 'black', color='black')

   dfARt = dfTrads[dfTrads['language'].str.startswith('ar')]
   dfARo = dfOrigs[dfOrigs['language'].str.startswith('ar')]
   
   dfCAt = dfTrads[dfTrads['language'].str.startswith('ca')]
   dfCAo = dfOrigs[dfOrigs['language'].str.startswith('ca')]
  
   dfRUt = dfTrads[dfTrads['language'].str.startswith('ru')]
   dfRUo = dfOrigs[dfOrigs['language'].str.startswith('ru')]

   dfTRt = dfTrads[dfTrads['language'].str.startswith('tr')]
   dfTRo = dfOrigs[dfOrigs['language'].str.startswith('tr')]
   
   dfHRt = dfTrads[dfTrads['language'].str.startswith('hr')]
   dfHRo = dfOrigs[dfOrigs['language'].str.startswith('hr')]
   
   dfESt = dfTrads[dfTrads['language'].str.startswith('es')]
   dfESo = dfOrigs[dfOrigs['language'].str.startswith('es')]

   dfDEt = dfTrads[dfTrads['language'].str.startswith('de')]
   dfDEo = dfOrigs[dfOrigs['language'].str.startswith('de')]

   dfITt = dfTrads[dfTrads['language'].str.startswith('it')]
   dfITo = dfOrigs[dfOrigs['language'].str.startswith('it')]

   dfENt = dfTrads[dfTrads['language'].str.startswith('en')]
   dfENo = dfOrigs[dfOrigs['language'].str.startswith('en')]


   if (weat=='tr'):
     plt.plot(dfARt['labels'], dfARt['sizeEffect'], 'm<', markersize=11, fillstyle='none')
     plt.plot(dfCAt['labels'], dfCAt['sizeEffect'], 'mv', markersize=11, fillstyle='none')
     plt.plot(dfRUt['labels'], dfRUt['sizeEffect'], 'm^', markersize=11, fillstyle='none')
     plt.plot(dfTRt['labels'], dfTRt['sizeEffect'], 'm>', markersize=11, fillstyle='none')
     plt.plot(dfHRt['labels'], dfHRt['sizeEffect'], 'yX', markersize=11, fillstyle='none')
     plt.plot(dfESt['labels'], dfESt['sizeEffect'], 'bp', markersize=11, fillstyle='none')
     plt.plot(dfDEt['labels'], dfDEt['sizeEffect'], 'rP', markersize=11, fillstyle='none')
     plt.plot(dfITt['labels'], dfITt['sizeEffect'], 'gs', markersize=11, fillstyle='none')
     plt.plot(dfENo['labels'], dfENo['effMedian'], 'ko', markersize=11)
     legend_elements = [
                   Line2D([0], [0], color='g', marker='s', label='it', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='r', marker='P', label='de', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='y', marker='X', label='hr', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='b', marker='p', label='es', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='m', marker='<', label='ar', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='m', marker='v', label='ca', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='m', marker='^', label='ru', markersize=15, fillstyle='none'),
                   Line2D([0], [0], color='m', marker='>', label='tr', markersize=15, fillstyle='none')
                   ,Line2D([0], [0], color='k', marker='o', label='en', markersize=15)
                   ]
   else:
     plt.plot(dfARo['labels'], dfARo['effMedian'], 'm<', markersize=11)
     plt.plot(dfCAo['labels'], dfCAo['effMedian'], 'mv', markersize=11)
     plt.plot(dfRUo['labels'], dfRUo['effMedian'], 'm^', markersize=11)
     plt.plot(dfTRo['labels'], dfTRo['effMedian'], 'm>', markersize=11)
     plt.plot(dfHRo['labels'], dfHRo['effMedian'], 'yX', markersize=11)
     plt.plot(dfESo['labels'], dfESo['effMedian'], 'bp', markersize=11)
     plt.plot(dfDEo['labels'], dfDEo['effMedian'], 'rP', markersize=11)
     plt.plot(dfITo['labels'], dfITo['effMedian'], 'gs', markersize=11)
     plt.plot(dfENo['labels'], dfENo['effMedian'], 'ko', markersize=11)
     legend_elements = [
                   Line2D([0], [0], color='g', marker='s', label='it', markersize=15),
                   Line2D([0], [0], color='r', marker='P', label='de', markersize=15),
                   Line2D([0], [0], color='y', marker='X', label='hr', markersize=15),
                   Line2D([0], [0], color='b', marker='p', label='es', markersize=15),
                   Line2D([0], [0], color='m', marker='<', label='ar', markersize=15),
                   Line2D([0], [0], color='m', marker='v', label='ca', markersize=15),
                   Line2D([0], [0], color='m', marker='^', label='ru', markersize=15),
                   Line2D([0], [0], color='m', marker='>', label='tr', markersize=15)
                   ,Line2D([0], [0], color='k', marker='o', label='en', markersize=15)
                   ]

   plt.ylabel('Effect size $d$', fontdict=fontAx)
   plt.ylim(-2.1,2)
   plt.yticks(np.arange(-2, 2.01, 1.0))
   plt.xticks(rotation=60)
   plt.tight_layout(pad=0.03)
   plt.legend(handles=legend_elements, ncol=5, loc='lower center', frameon=False)
   #plt.show()
   if (weat=='tr'):
      plt.savefig('effectSize_WEAT'+str(test)+'TR.png')
   else:
      plt.savefig('effectSize_WEAT'+str(test)+'OR.png')
   plt.clf()
    
    
    
    
def plotIsomorf(df, measure, test):
   #model = 'cc10017vecmapSup'
   #dfWithISO = df[(df[measure].str.contains('\.')) & (df['test']==test) & (df['source']==0)  & (df['model']==model) ]
   dfWithISO = df[(df[measure].str.contains('\.')) & (df['test']==test)]
   #dfWithISO = df[(df[measure].str.contains('\.')) & (df['test']==test) & (df['source']==1)]
 #  dfWithISO = df[(df[measure].str.contains('\.')) & (df['test']==test) & ((df['model'].str.startswith('w')) | (df['model'].str.startswith('x')) | (df['model'].str.startswith('b')))  ]
#   dfWithISO = df[(df[measure].str.isnumeric()) & (df['test']==test) & (df['source']==1) ]
   asymmetric_error = np.array(list(zip(dfWithISO['DsizeEffectLO'], dfWithISO['DsizeEffectUP']))).T

   mb, stats = poly.polyfit(dfWithISO[measure].astype('float'), dfWithISO['sizeEffect'], 1, full=True)
   SSE = stats[0][0]
   diff2 = (dfWithISO['sizeEffect'] - dfWithISO['sizeEffect'].mean())**2
   SST = diff2.sum()
   r2 = 1-SSE/SST 
   print('r2=',r2)

   fig = plt.figure(figsize=(9.5, 5))
   plt.errorbar(dfWithISO[measure].astype('float'), dfWithISO['sizeEffect'], yerr=asymmetric_error, fmt='none', ecolor = 'darkred', color='lightslategrey')
   print(dfWithISO[measure].astype('float').corr(dfWithISO['sizeEffect']))

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('ar'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'm<', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('hr'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'yX', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('it'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'gs', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('es'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'bp', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('de'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'rP', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('ca'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'mv', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('ru'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'm^', markersize=11)

   dfLan = dfWithISO[(dfWithISO['language'].str.startswith('tr'))]
   plt.plot(dfLan[measure].astype('float'),  dfLan['sizeEffect'], 'm>', markersize=11)

   m,b = np.polyfit(dfWithISO[measure].astype('float'), dfWithISO['sizeEffect'].astype('float'), 1)
   plt.plot(dfWithISO[measure].astype('float'), m*dfWithISO[measure].astype('float') + b)

   legend_elements = [
                   Line2D([0], [0], color='g', marker='s', label='it', markersize=17),
                   Line2D([0], [0], color='r', marker='P', label='de', markersize=17),
                   Line2D([0], [0], color='y', marker='X', label='hr', markersize=17),
                   Line2D([0], [0], color='b', marker='p', label='es', markersize=17),
                   Line2D([0], [0], color='m', marker='<', label='ar', markersize=17),
                   Line2D([0], [0], color='m', marker='v', label='ca', markersize=17),
                   Line2D([0], [0], color='m', marker='^', label='ru', markersize=17),
                   Line2D([0], [0], color='m', marker='>', label='tr', markersize=17)
   #                ,Line2D([0], [0], color='k', marker='o', label='en', markersize=17)
                   ]

   
   plt.ylabel('Effect size $d$', fontdict=fontAx)
   plt.xlabel(measure, fontdict=fontAx)
   if (measure=='EV'):
       plt.axis([0, 500, -2, 2])
   elif (measure=='GH'):
       plt.axis([0, 2.5, -2, 2])
   else:
       print("Isometry measure not implemented")
   plt.yticks(np.arange(-2, 2.01, 1.0))
   #plt.title('    CA-WEAT '+str(test), fontdict=fontTit)
   plt.tight_layout(pad=0.03)
   plt.legend(handles=legend_elements, ncol=4, loc='lower center', frameon=False)
   #plt.show()
   plt.savefig('iso'+measure+'_allWEAT'+str(test)+'.png')
   plt.clf()



def plotEVvsGH(df):
#   dfWithISO = df[(df['EV'].str.contains('\.')) & (df['test']==1) & (df['source']==0)  ]
   dfWithISO = df[(df['EV'].str.contains('\.')) & (df['test']==1) ]

   fig = plt.figure(figsize=(6.5, 6))
   plt.plot(dfWithISO['EV'].astype('float'), dfWithISO['GH'].astype('float'), marker='o', lw=0, color='lightslategrey')

   m,b = np.polyfit(dfWithISO['EV'].astype('float'), dfWithISO['GH'].astype('float'), 1)
   plt.plot(dfWithISO['EV'].astype('float'), m*dfWithISO['EV'].astype('float') + b)
   
   print(dfWithISO['EV'].astype('float').corr(dfWithISO['GH'].astype('float')))
   print(len(dfWithISO['EV'].astype('float')))
   plt.ylabel('GH', fontdict=fontAx)
   plt.xlabel('EV', fontdict=fontAx)
   plt.axis([0, 500, 0, 2.5])
   #plt.yticks(np.arange(-2, 2.01, 1.0))
   #plt.title('    CA-WEAT '+str(test), fontdict=fontTit)
   plt.tight_layout(pad=0.03)
   plt.show()
   # plt.savefig(outputPlotF)
   plt.clf()


   
def caweatBITable(df):
   languages = ["en", "ar", "ca", "de", "es", "hr", "it", "ru", "tr"]
   for test in [1,2]:
       if(test == 1):
          print('\mc{6}{l}{\\normalsize \it WEAT 1: Flowers and insects} \\\\')
       if(test == 2):
          print('\midrule')
          print('\mc{6}{l}{\\normalsize \it WEAT 2: Instruments and weapons} \\\\')
       for model in models:
           name = naming[model]
           # median of the original English which is stored as translation, but it's an original
           if(model.endswith('XX')):
              model = model.replace('XX','en')
           minidf = df[(df['model']==model) & (df['test']==test) & (df['language']=='en'+str(0))]
           if(model.endswith('en')):
              model = model.replace('en','XX')
           if minidf.empty:
              statEffOrig = '-- &'
           else:   
              statTrad = round(minidf['statistic'].astype('float'),1).values[0]
              statTradU = round(minidf['DstatUP'].astype('float'),1).values[0]
              statTradD = round(minidf['DstatLO'].astype('float'),1).values[0]
              #statEffOrig = '$'+str(statTrad)+'^{+'+str(statTradU)+'}'+'_{-'+str(statTradD)+'}$ &'
              statEffOrig = '$'+str(statTrad)+'$ &'

           line = statEffOrig
           for lan in languages:   
               if(model.endswith('XX')):
                  model = model.replace('XX',lan)
               # median of the originals
               if(lan=='es'):
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'_ES1')]
               elif(lan=='en'):
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'_US1')]
               else:
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'1')]               
               if minidf.empty:
                  statEffOrig = ' -- &'
               else:
                  medOrig = round(minidf['statMedian'].astype('float'),1).values[0]
                  medOrigU = round(minidf['DstatMedUP'].astype('float'),1).values[0]
                  medOrigD = round(minidf['DstatMedLO'].astype('float'),1).values[0]
                  if (lan=='ar'): 
                      statEffOrig = ' $'+str(medOrig)+'$ &'
                  else:
                      statEffOrig = ' $'+str(medOrig)+'^{+'+str(medOrigU)+'}'+'_{-'+str(medOrigD)+'}$ &'
               # a language is done
               line = line + statEffOrig
               if(model.endswith(lan)):
                  model = model.replace(lan,'XX')
           print(name + ' & ' +line[:-1] + ' \\\\')
               
def xweatBITable(df):
   languages = ["ar", "ca", "de", "es", "hr", "it", "ru", "tr"]
   for test in [1,2]:
       if(test == 1):
          print('\mc{6}{l}{\\normalsize \it WEAT 1: Flowers and insects} \\\\')
       if(test == 2):
          print('\midrule')
          print('\mc{10}{l}{\\normalsize \it WEAT 2: Instruments and weapons} \\\\')
       for model in models:
           name = naming[model]
           # median of the original English which is stored as translation, but it's an original
           if(model.endswith('XX')):
              model = model.replace('XX','en')
           minidf = df[(df['model']==model) & (df['test']==test) & (df['language']=='en'+str(0))]
           if(model.endswith('en')):
              model = model.replace('en','XX')
           if minidf.empty:
              statEffOrig = '-- &'
           else:
              statTrad = round(minidf['statistic'].astype('float'),1).values[0]
              statTradU = round(minidf['DstatUP'].astype('float'),1).values[0]
              statTradD = round(minidf['DstatLO'].astype('float'),1).values[0]
              statEffOrig = '$'+str(statTrad)+'^{+'+str(statTradU)+'}'+'_{-'+str(statTradD)+'}$ &'

           line = statEffOrig
           for lan in languages:   
               if(model.endswith('XX')):
                  model = model.replace('XX',lan)
               # translation (bootstrapped CI)
               minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+str(0))]
               if minidf.empty:
                  statEffTrad = ' -- &'
               else:
                  statTrad = round(minidf['statistic'].astype('float'),1).values[0]
                  statTradU = round(minidf['DstatUP'].astype('float'),1).values[0]
                  statTradD = round(minidf['DstatLO'].astype('float'),1).values[0]
                  statEffTrad = ' $'+str(statTrad)+'^{+'+str(statTradU)+'}'+'_{-'+str(statTradD)+'}$ &'
               # a language is done
               line = line + statEffTrad 
               if(model.endswith(lan)):
                  model = model.replace(lan,'XX')
           print(name + ' & ' +line[:-1] + ' \\\\')

   
def caweatESTable(df):
   languages = ["en", "ar", "ca", "de", "es", "hr", "it", "ru", "tr"]
   for test in [1,2]:
       if(test == 1):
          print('\mc{6}{l}{\\normalsize \it WEAT 1: Flowers and insects} \\\\')
       if(test == 2):
          print('\midrule')
          print('\mc{6}{l}{\\normalsize \it WEAT 2: Instruments and weapons} \\\\')
       for model in models:
           name = naming[model]
           # median of the original English which is stored as translation, but it's an original
           if(model.endswith('XX')):
              model = model.replace('XX','en')
           minidf = df[(df['model']==model) & (df['test']==test) & (df['language']=='en'+str(0))]
           if(model.endswith('en')):
              model = model.replace('en','XX')
           if minidf.empty:
              sizeEffOrig = '-- &'
           else:
              sizeTrad = round(minidf['sizeEffect'].astype('float'),1).values[0]
              sizeTradU = round(minidf['DsizeEffectUP'].astype('float'),1).values[0]
              sizeTradD = round(minidf['DsizeEffectLO'].astype('float'),1).values[0]
              #sizeEffOrig = '$'+str(sizeTrad)+'^{+'+str(sizeTradU)+'}'+'_{-'+str(sizeTradD)+'}$ &'
              sizeEffOrig = '$'+str(sizeTrad)+'$ &'

           line = sizeEffOrig
           for lan in languages:   
               if(model.endswith('XX')):
                  model = model.replace('XX',lan)
               # median of the originals
               if(lan=='es'):
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'_ES1')]
               elif(lan=='en'):
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'_US1')]
               else:
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'1')]               
               if minidf.empty:
                  sizeEffOrig = ' -- &'
               else:
                  medOrig = round(minidf['effMedian'].astype('float'),1).values[0]
                  medOrigU = round(minidf['DeffMedUP'].astype('float'),1).values[0]
                  medOrigD = round(minidf['DeffMedLO'].astype('float'),1).values[0]
                  if (lan=='ar'): 
                      sizeEffOrig = ' $'+str(medOrig)+'$ &'
                  else:
                      sizeEffOrig = ' $'+str(medOrig)+'^{+'+str(medOrigU)+'}'+'_{-'+str(medOrigD)+'}$ &'
               # a language is done
               line = line + sizeEffOrig
               if(model.endswith(lan)):
                  model = model.replace(lan,'XX')
           print(name + ' & ' +line[:-1] + ' \\\\')
               
def xweatESTable(df):
   languages = ["ar", "ca", "de", "es", "hr", "it", "ru", "tr"]
   for test in [1,2]:
       if(test == 1):
          print('\mc{6}{l}{\\normalsize \it WEAT 1: Flowers and insects} \\\\')
       if(test == 2):
          print('\midrule')
          print('\mc{10}{l}{\\normalsize \it WEAT 2: Instruments and weapons} \\\\')
       for model in models:
           name = naming[model]
           # median of the original English which is stored as translation, but it's an original
           if(model.endswith('XX')):
              model = model.replace('XX','en')
           minidf = df[(df['model']==model) & (df['test']==test) & (df['language']=='en'+str(0))]
           if(model.endswith('en')):
              model = model.replace('en','XX')
           if minidf.empty:
              sizeEffOrig = '-- &'
           else:
              sizeTrad = round(minidf['sizeEffect'].astype('float'),1).values[0]
              sizeTradU = round(minidf['DsizeEffectUP'].astype('float'),1).values[0]
              sizeTradD = round(minidf['DsizeEffectLO'].astype('float'),1).values[0]
              sizeEffOrig = '$'+str(sizeTrad)+'^{+'+str(sizeTradU)+'}'+'_{-'+str(sizeTradD)+'}$ &'

           line = sizeEffOrig
           for lan in languages:   
               if(model.endswith('XX')):
                  model = model.replace('XX',lan)
               # translation (bootstrapped CI)
               minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+str(0))]
               if minidf.empty:
                  sizeEffTrad = ' -- &'
               else:
                  sizeTrad = round(minidf['sizeEffect'].astype('float'),1).values[0]
                  sizeTradU = round(minidf['DsizeEffectUP'].astype('float'),1).values[0]
                  sizeTradD = round(minidf['DsizeEffectLO'].astype('float'),1).values[0]
                  sizeEffTrad = ' $'+str(sizeTrad)+'^{+'+str(sizeTradU)+'}'+'_{-'+str(sizeTradD)+'}$ &'
               # a language is done
               line = line + sizeEffTrad 
               if(model.endswith(lan)):
                  model = model.replace(lan,'XX')
           print(name + ' & ' +line[:-1] + ' \\\\')

def mainOldTable(df):
   languages = ["ar", "ca", "de", "es", "hr", "it", "ru", "tr"]
   #languages = ["ar", "it"]
   for test in [1,2]:
       if(test == 1):
          print('\mc{10}{l}{\Large \it WEAT 1: Flowers and insects} \\\\')
       if(test == 2):
          print('\midrule')
          print('\mc{10}{l}{\Large \it WEAT 2: Instruments and weapons} \\\\')
       for model in models:
           name = naming[model]
           # median of the original English which is stored as translation, but it's an original
           if(model.endswith('XX')):
              model = model.replace('XX','en')
           minidf = df[(df['model']==model) & (df['test']==test) & (df['language']=='en'+str(0))]
           if(model.endswith('en')):
              model = model.replace('en','XX')
           if minidf.empty:
              sizeEffOrig = '-- &'
           else:
              sizeTrad = round(minidf['sizeEffect'].astype('float'),1).values[0]
              sizeTradU = round(minidf['DsizeEffectUP'].astype('float'),1).values[0]
              sizeTradD = round(minidf['DsizeEffectLO'].astype('float'),1).values[0]
              sizeEffOrig = '$'+str(sizeTrad)+'^{+'+str(sizeTradU)+'}'+'_{-'+str(sizeTradD)+'}$ &'

           line = sizeEffOrig
           for lan in languages:   
               if(model.endswith('XX')):
                  model = model.replace('XX',lan)
               # median of the originals
               if(lan=='es'):
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'_ES1')]
               else:
                  minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+'1')]               
               if minidf.empty:
                  sizeEffOrig = ' -- &'
               else:
                  medOrig = round(minidf['effMedian'].astype('float'),1).values[0]
                  medOrigU = round(minidf['DeffMedUP'].astype('float'),1).values[0]
                  medOrigD = round(minidf['DeffMedLO'].astype('float'),1).values[0]
                  if (lan=='ar'): 
                      sizeEffOrig = ' $'+str(medOrig)+'$ &'
                  else:
                      sizeEffOrig = ' $'+str(medOrig)+'^{+'+str(medOrigU)+'}'+'_{-'+str(medOrigD)+'}$ &'
               # translation (bootstrapped CI)
               minidf = df[(df['model']==model) & (df['test']==test) & (df['language']==lan+str(0))]
               if minidf.empty:
                  sizeEffTrad = ' -- &'
               else:
                  sizeTrad = round(minidf['sizeEffect'].astype('float'),1).values[0]
                  sizeTradU = round(minidf['DsizeEffectUP'].astype('float'),1).values[0]
                  sizeTradD = round(minidf['DsizeEffectLO'].astype('float'),1).values[0]
                  sizeEffTrad = ' $'+str(sizeTrad)+'^{+'+str(sizeTradU)+'}'+'_{-'+str(sizeTradD)+'}$ &'
               # a language is done
               line = line + sizeEffOrig + sizeEffTrad
               if(model.endswith(lan)):
                  model = model.replace(lan,'XX')
           print(name + ' & ' +line[:-1] + ' \\\\')
               
               
# Reading the data
df = pd.read_csv(dataFile) 

#plotCountsXweat(df, 1)
#plotCounts(df, 1)
#plotCountsXweat(df, 2)
#plotCounts(df, 2)

#plotSExModel(df, 1, 'tr')
#plotSExModel(df, 2, 'tr')
#plotSExModel(df, 1, 'or')
#plotSExModel(df, 2, 'or')

#plotIsomorf(df, 'EV', 1)
#plotIsomorf(df, 'EV', 2)
plotIsomorf(df, 'GH', 1)
#plotIsomorf(df, 'GH', 2)

#plotEVvsGH(df)

#caweatBITable(df)
#xweatBITable(df)
#caweatESTable(df)
#xweatESTable(df)
#mainOldTable(df)

