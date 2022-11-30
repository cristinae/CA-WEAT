# CA-WEAT, Cultural Aware Word Embedding Association Tests

This repository contains the data and scripts necessary to reproduce the experiments in 

Cristina España-Bonet and Alberto Barrón-Cedeño. 2022. The (Undesired) Attenuation of Human Biases by Multilinguality. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Online and Abu Dhabi. Association for Computational Linguistics.

```
@inproceedings{espana-bonet-etal-2022-attenuation,
    title = "Comparing Feature-Engineering and Feature-Learning Approaches for Multilingual Translationese Classification",
    author = "Espa{\~n}a-Bonet, Cristina  and  Barrón-Cede{\~n}o, Alberto",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Online and Abu Dhabi",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.",
    doi = "",
    pages = "--"
}
```

## CA-WEATs

The folder ```data``` contains the tsv files with the cultural aware lists for WEAT1 and WEAT2 in 26 languages 
(```ar```, 
```bg```, 
```bn```, 
```ca```, 
```de```, 
```el```, 
```en```, 
```es```, 
```fa```, 
```fr```, 
```hr```, 
```id```, 
```it```, 
```ko```, 
```lb```, 
```mr```, 
```nl```, 
```no```, 
```pl```, 
```pt```, 
```ro```, 
```ru```, 
```tr```, 
```uk```, 
```vi```, 
```zh```).

<p align="center">
  <img src="data/CA_WEATv1s.png" width="1100" title="Distribution per country">
</p>

## Bias and Size Effect

The calculation of the statistic and size effect has been adapted from [Lauscher and Glavas (SEM* 2019)](https://github.com/umanlp/XWEAT). Please, use ```runCaweat.sh``` specifying the languages to consider (LANGID in the [CA-WEAT file](data/CA-WEATv1.tsv)) and the embedding model. Feel free to change the number of permutations to calculate p-values or the number of bootstraps for confidence intervals. If your embedding model has the vocabulary lowercased use the option ```--lower```.

### Tables and Plots

Results for the 16 embedding models and the 91 lists used in the paper are collected in ```plots/collectedData.csv```  and the script ```plots/plotCollectedData.py``` can be used to generate the plots and tables in a straightforward manner. 
