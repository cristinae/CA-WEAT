Script `caweat_inspector.py` identifies records that lack a full entry (e.g., missing flowers) as well as those where a term is duplicated.

Running examples:

```

$ python3 caweat_inspector.py -l Greek -t CulturalAwareWEAT-tmp.tsv

$ python3 caweat_inspector.py -l Italian -t CulturalAwareWEAT-it.tsv

```

If this process triggers an error, it is likely that a full entry is empty. In order to find it, run the script with flag -e:

```
$ python3 caweat_inspector.py -l Italian -t CulturalAwareWEAT-it.tsv -e
```

Script `caweat_completer.py` identifies records that lack a full entry (e.g., missing flowers) as well as those where a term is duplicated, and recommends the next item per volunteer using matrix factorisation.

Running examples:

```
$ python3 caweat_completer.py -c flowers -l Spanish -t CulturalAwareWEAT-es.tsv 

$ python3 caweat_completer.py -c unpleasant -l English -t CulturalAwareWEAT-el.tsv 
```

Script `caweat_jsoner.py` converts the tsv into a json file and vice-versa:

```
# From tsv to json
$ python3 caweat_jsoner.py -i CA-WEATv3.tsv -o CA-WEATv3.json

# From json to tsv
$ python3 caweat_jsoner.py -t -i  CA-WEATv3.json -o CA-WEATv3.tsv
```

