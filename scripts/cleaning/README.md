Script `caweat_inspector.py identifies records that lack a full entry (e.g., missing fruits) as well as those where a term is duplicated.

Running examples:

```

$ python caweat_inspector.py -l Greek -t CulturalAwareWEAT-tmp.tsv

$ python caweat_inspector.py -l Italian -t CulturalAwareWEAT-it.tsv

```

If this process triggers an error, it is likely that a full entry is empty. In order to find it, run the script with flag -e:

```
$ python caweat_inspector.py -l Italian -t CulturalAwareWEAT-it.tsv -e
```
