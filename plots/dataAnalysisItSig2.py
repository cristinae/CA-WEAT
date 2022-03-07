import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator
import re
import numpy as np

prefixP = './it/pValues'
prefixS = './it/sigmas'
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
'bert0it', 
'bert11it',
'xlmr0',
'xlmr11']

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
        'size': 22,
        }
fontAx = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 20,
        }

# Loading counts
with open ('itListsComplete.stats', 'rt') as statsFile: 
     contents = statsFile.readlines()
countsPlUnPl = []
countsPl = []
countsUn = []
restaPlUn = []
countsW1 = []
countsW2 = []
for line in contents:
    tmp = re.findall(r',\s*(\d+)', line.strip())
    countsPlUnPl.append(int(tmp[0]))
    countsW1.append(int(tmp[1]))
    countsW2.append(int(tmp[2]))
    countsPl.append(int(tmp[3]))
    countsUn.append(int(tmp[4]))
    restaPlUn.append(int(tmp[3])-int(tmp[4]))
    
# Loading images
imageFolder = './it_plots2/img/'
imgInstrument = mpimg.imread(imageFolder+'62809-guitar-icon.png')
imgWeapon = mpimg.imread(imageFolder+'62964-pistol-icon.png')
imgInsect = mpimg.imread(imageFolder+'dragon-fly-icon.png')
imgFlower = mpimg.imread(imageFolder+'flower-icon.png')
imgUnpl = mpimg.imread(imageFolder+'big-smile-icon.png')
imgPlea = mpimg.imread(imageFolder+'unhappy-icon.png')

for test in range(1,3):
  for model in models:
    statistic = []
    sizeEffect = []
    statisticSig = []
    statisticSigUp = []
    statisticSigLo = []
    sizeEffectSig = []   
    sizeEffectSigUp = []   
    sizeEffectSigLo = []   
    labels = []
    print(model)
    outputPlot1 = './it_plots2/caMix_'+model+'_its_cosine_'+str(test)+'_statistic.png'
    outputPlot2 = './it_plots2/caMix_'+model+'_its_cosine_'+str(test)+'_sizeff.png'
    outputPlotF = './it_plots2/freqs/ca_'+model+'_its_cosine_'+str(test)+'_freq.png'
    label = 1
    for i in range(1,24):
        #fileName = prefixP+'/ca_'+model+'_it' +str(i)+'_cosine_'+str(test)+'_uncased.res'
        fileNameSig = prefixS+'/caMix_'+model+'_it' +str(i)+'_cosine_'+str(test)+'_uncased.res'
        res = []
        means = []
        sigmas = []
        # we read the results with the original lists for medians
        # WE DONT REALLY NEED THIS, ALL THE INFO BUT P-VALUES ARE IN THE OTHER FILE TOO
        #try:
        #  with open (fileName, 'rt') as resultsFile: 
        #     contents = resultsFile.read() 
        #     res = re.findall(r'-*[0|1]\.\d\d\d\d\d\d\d\d\d', contents)
        #except FileNotFoundError:
        #   print("skipping it" + str(i))
        #   continue
        #statistic.append(float(res[0]))
        #sizeEffect.append(float(res[1]))

        # we read the results with boostrapped lists
        try:
          with open (fileNameSig, 'rt') as resultsFile: 
             contents2 = resultsFile.read() 
             res = re.findall(r'-*[0|1|2]\.\d\d\d\d\d\d\d\d\d', contents2)
             means = re.findall(r'(-*\d\.\d\d)\$', contents2)
             sigmas = re.findall(r'([0|1|2]\.\d\d)\}', contents2)
        except FileNotFoundError:
           print("skipping it" + str(i))
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

    # we read the results with the translated boostrapped list
    fileNameSigTrad = prefixS+'/trads/ca_'+model+'_it_cosine_'+str(test)+'_uncased.res'
    with open (fileNameSigTrad, 'rt') as resultsFile: 
        contents = resultsFile.read() 
    meansTrad = re.findall(r'(-*\d\.\d\d)\$', contents)
    sigmasTrad = re.findall(r'([0|1|2]\.\d\d)\}', contents)
  
    # Preparing for the statistic plot
    median = np.percentile(statisticSig,50)
    ci_high = np.percentile(statisticSig,95)
    ci_low = np.percentile(statisticSig,5)
    max_ci = '{:.2f}'.format(round(ci_high-median,2))
    min_ci = '{:.2f}'.format(round(median-ci_low,2))
    medianRes = '$'+'{:.2f}'.format(round(median,2))+ '^{+' + max_ci + '}_{-' + min_ci +'}$'
    transRes = '$'+meansTrad[0]+ '^{+' + sigmasTrad[0] + '}_{-' + sigmasTrad[1] +'}$'
    #plt.hist(statistic, bins = 7, color='lightslategrey')
    #plt.ylabel('\# of instances', fontdict=fontAx)
    #plt.xlabel('Statistic', fontdict=fontAx)
    #plt.axis([-0.25, 2, 0, 10])
    #plt.title(model+'    CA-WEAT'+str(test)+',   median: '+medianRes, fontdict=fontTit)
    #plt.savefig(outputPlot1)
        
    #figure(figsize=(8, 6), dpi=80)
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, hspace=0,height_ratios=[1.3,1])
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(model+'    CAmix-WEAT '+str(test), fontdict=fontTit)
    axs[0].hist(statistic, bins = 7, color='lightslategrey')
#    axs[0].text(0.78, 0.90,  'median: '+medianRes, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].text(0.04, 0.90,  'median: '+medianRes, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, color='dimgrey')
    axs[0].text(0.04, 0.75,  'trans: '+transRes, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, color='darkblue')
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    asymmetric_error = np.array(list(zip(statisticSigLo, statisticSigUp))).T
    axs[1].errorbar(statisticSig, labels, xerr=asymmetric_error, fmt='o', ecolor = 'darkred', color='lightslategrey')
    plt.xlim(-0.5, 2)
    axs[1].errorbar(float(meansTrad[0]), -1, xerr=[[float(sigmasTrad[1])],[float(sigmasTrad[0])]], fmt='o', ecolor = 'darkblue', color='darkblue', ls='--')
    axs[0].set_yticks(np.arange(0,10.5,2))
    axs[1].set_ylim(-3,24)
    axs[1].set_xlabel('Statistic', fontdict=fontAx)
    axs[0].set_ylabel('\# of instances', fontdict=fontAx)
    axs[1].set_ylabel('orig. instance \#', fontdict=fontAx)
    fig.align_ylabels(axs)
    for ax in axs:
        ax.label_outer()
    plt.tight_layout(pad=0.3)
    #plt.show()
    plt.savefig(outputPlot1)
    plt.clf()

    
    # Preparing for the size effect plot
#    print(sizeEffect)
    median = np.percentile(sizeEffect,50)
    ci_high = np.percentile(sizeEffect,95)
    ci_low = np.percentile(sizeEffect,5)
    max_ci = '{:.2f}'.format(round(ci_high-median,2))
    min_ci = '{:.2f}'.format(round(median-ci_low,2))
    medianRes = '$'+'{:.2f}'.format(round(median,2))+ '^{+' + max_ci + '}_{-' + min_ci +'}$'
    transRes = '$'+meansTrad[1]+ '^{+' + sigmasTrad[2] + '}_{-' + sigmasTrad[3] +'}$'
    #plt.hist(sizeEffect, bins = 7, color='lightslategrey')
    #plt.ylabel('\# of instances', fontdict=fontAx)
    #plt.xlabel('Size effect', fontdict=fontAx)
    #plt.axis([-2, 2, 0, 10])
    #plt.title(model+'    CA-WEAT'+str(test)+',   median: '+medianRes, fontdict=fontTit)
    
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, hspace=0,height_ratios=[1.3,1])
    axs = gs.subplots(sharex=True, sharey=False)
#    fig.suptitle(model+'    CA-WEAT'+str(test)+',   median: '+medianRes, fontdict=fontTit)
    fig.suptitle(model+'    CAmix-WEAT '+str(test), fontdict=fontTit)
    axs[0].hist(sizeEffect, bins = 7, color='lightslategrey')
    axs[0].text(0.04, 0.90,  'median: '+medianRes, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, color='dimgrey')
    axs[0].text(0.04, 0.75,  'trans: '+transRes, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, color='darkblue')
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    asymmetric_error = np.array(list(zip(sizeEffectSigLo, sizeEffectSigUp))).T
    axs[1].errorbar(sizeEffectSig, labels, xerr=asymmetric_error, fmt='o', ecolor = 'darkred', color='lightslategrey')
    axs[1].errorbar(float(meansTrad[1]), -1, xerr=[[float(sigmasTrad[3])],[float(sigmasTrad[2])]], fmt='o', ecolor = 'darkblue', color='darkblue', ls='--')
    axs[0].set_yticks(np.arange(0,10.5,2))
    plt.xlim(-2, 2)
#    axs[0].set_ylim(0,10)
    axs[1].set_ylim(-3,24)
    axs[1].set_xlabel('Size effect', fontdict=fontAx)
    axs[0].set_ylabel('\# of instances', fontdict=fontAx)
    axs[1].set_ylabel('orig. instance \#', fontdict=fontAx)
    fig.align_ylabels(axs)
    for ax in axs:
        ax.label_outer()    
    plt.tight_layout(pad=0.3)
    plt.savefig(outputPlot2)
    plt.clf()

    # Ploting size effect with respect to the number of times the words in the lists appear in the corpus
    if(model.startswith('cc10017') or model.startswith('w2v')):    
       countsAll=[]
       countsAtt=[]
       if(test==1):
          countsAtt = restaPlUn[:22]
          countsAll = [sum(x) for x in zip(countsW1[:22], countsPlUnPl[:22])]
       elif(test==2):
          countsAtt =  restaPlUn[:22]
          countsAll = [sum(x) for x in zip(countsW2[:22], countsPlUnPl[:22])]
       fig = plt.figure(figsize=(6.5, 6))

       #plt.errorbar(countsAll, sizeEffectSig, yerr=asymmetric_error, fmt='o', ecolor = 'darkred', color='lightslategrey')
       #m,b = np.polyfit(countsAll, sizeEffectSig, 1)
       #plt.plot(np.array(countsAll), m*np.array(countsAll) + b)

       plt.errorbar(countsAtt, sizeEffectSig, yerr=asymmetric_error, fmt='o', ecolor = 'darkred', color='lightslategrey')
       m,b = np.polyfit(countsAtt, sizeEffectSig, 1)
       plt.plot(np.array(countsAtt), m*np.array(countsAtt) + b)
       
       #plt.imshow(imgUnpl, extent=(2200000,3100000,-1.5,-1.1), aspect='auto')
       #plt.imshow(imgPlea, extent=(3200000,4500000,-1.5,-1.1), aspect='auto')
       #if(test==1):
       #  plt.imshow(imgFlower, extent=(4600000,6600000,-1.5,-1.1), aspect='auto')
       #  plt.imshow(imgInsect, extent=(6700000,9000000,-1.5,-1.1), aspect='auto')
       #  plt.imshow(imgFlower, extent=(240000,350000,-1.5,-1.1), aspect='auto')
       #  plt.imshow(imgInsect, extent=(370000,500000,-1.5,-1.1), aspect='auto')
       #elif(test==2):
       #  plt.imshow(imgInstrument, extent=(4600000,6500000,-1.5,-1.1), aspect='auto')
       #  plt.imshow(imgWeapon, extent=(6700000,9600000,-1.55,-1.05), aspect='auto')
       #  plt.imshow(imgInstrument, extent=(400000,580000,-1.5,-1.1), aspect='auto')
       #  plt.imshow(imgWeapon, extent=(600000,880000,-1.55,-1.05), aspect='auto')

       #plt.xscale('log')
       plt.ylabel('Size effect', fontdict=fontAx)
       plt.xlabel('$\Delta$Counts (Pleasant-Unpleasant)', fontdict=fontAx)
       plt.axis([-1000000, 5000000, -2, 2])
       #plt.xticks(np.arange(200000, 10000000, 300000))
       plt.yticks(np.arange(-2, 2.01, 1.0))
       plt.title(model+'    CA-WEAT '+str(test), fontdict=fontTit)
       plt.tight_layout(pad=0.03)
       plt.savefig(outputPlotF)
       plt.clf()
       #plt.show()
    

