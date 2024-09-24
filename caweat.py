import numpy as np
import random
from itertools import filterfalse
from itertools import combinations
import codecs
import unidecode
import pandas as pd
import utils
import os
import pickle
import logging
import argparse
import time
from collections import OrderedDict
import math
from sklearn.metrics.pairwise import euclidean_distances



class CAWEAT(object):
  """
  Perform WEAT (Word Embedding Association Test) bias tests on embeddings.
  Follows from Caliskan et al 2017 (10.1126/science.aal4230).

  Adapted from Lauscher A. and Glavas G. (SEM* 2019). 
  https://github.com/anlausch/XWEAT
  Credits: Basic implementation based on https://gist.github.com/SandyRogers/e5c2e938502a75dcae25216e4fae2da5
  """

  def __init__(self):
    self.embd_dict = None
    self.vocab = None
    self.embedding_matrix = None

  def set_embd_dict(self, embd_dict):
    self.embd_dict = embd_dict


  def _build_vocab_dict(self, vocab):
    self.vocab = OrderedDict()
    vocab = set(vocab)
    index = 0
    for term in vocab:
      if term in self.embd_dict:
        self.vocab[term] = index
        index += 1
      else:
        logging.warning("Not in vocab %s", term)


  def convert_by_vocab(self, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
      if item in self.vocab:
        output.append(self.vocab[item])
      else:
        continue
    return output

  def _build_embedding_matrix(self):
    self.embedding_matrix = []
    for term, index in self.vocab.items():
      if term in self.embd_dict:
        self.embedding_matrix.append(self.embd_dict[term])
      else:
        raise AssertionError("This should not happen.")
    self.embd_dict = None


  def mat_normalize(self,mat, norm_order=2, axis=1):
    return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])


  def cosine(self, a, b):
    norm_a = self.mat_normalize(a)
    norm_b = self.mat_normalize(b)
    cos = np.dot(norm_a, np.transpose(norm_b))
    return cos


  def euclidean(self, a, b):
    norm_a = self.mat_normalize(a)
    norm_b = self.mat_normalize(b)
    distances = euclidean_distances(norm_a, norm_b)
    eucl = 1/ (1+distances)
    return eucl


  def csls(self, a, b, k=10):
    norm_a = self.mat_normalize(a)
    norm_b = self.mat_normalize(b)
    sims_local_a = np.dot(norm_a, np.transpose(norm_a))
    sims_local_b = np.dot(norm_b, np.transpose(norm_b))

    csls_norms_a = np.mean(np.sort(sims_local_a, axis=1)[:, -k - 1:-1], axis=1)
    csls_norms_b = np.mean(np.sort(sims_local_b, axis=1)[:, -k - 1:-1], axis=1)
    loc_sims = np.add(np.transpose(np.tile(csls_norms_a, (len(csls_norms_b), 1))),
                      np.tile(csls_norms_b, (len(csls_norms_a), 1)))

    return 2 * np.dot(norm_a, np.transpose(norm_b)) - loc_sims


  def _init_similarities(self, similarity_type):
    if similarity_type == "cosine":
      self.similarities = self.cosine(self.embedding_matrix, self.embedding_matrix)
    elif similarity_type == "csls":
      self.similarities = self.csls(self.embedding_matrix, self.embedding_matrix)
    elif similarity_type == "euclidean":
      self.similarities = self.euclidean(self.embedding_matrix, self.embedding_matrix)
    else:
      raise NotImplementedError()


  def similarity_precomputed_sims(self, w1, w2, type="cosine"):
    return self.similarities[w1, w2]


  def word_association_with_attribute_precomputed_sims(self, w, A, B):
    return np.mean([self.similarity_precomputed_sims(w, a) for a in A]) - np.mean([self.similarity_precomputed_sims(w, b) for b in B])


  def differential_association_precomputed_sims(self, T1, T2, A1, A2):
    return np.sum([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) \
           - np.sum([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])

  def weat_effect_size_precomputed_sims(self, T1, T2, A1, A2):
    return (
             np.mean([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) -
             np.mean([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])
           ) / np.std([self.word_association_with_attribute_precomputed_sims(w, A1, A2) for w in T1 + T2])


  def _generate_bootstrap(self, V):
    bootstrap =random.choices(V, k=len(V))
    return bootstrap

  def sigma_differential_association_precomputed_sims(self, T1, T2, A1, A2, samples, confidence_level):
    differentials = np.zeros(samples)
    for i in range(samples):
        differentials[i] = self.differential_association_precomputed_sims(self._generate_bootstrap(T1), self._generate_bootstrap(T2), self._generate_bootstrap(A1), self._generate_bootstrap((A2)))
    return np.percentile(differentials,[(100-confidence_level)/2,(100-(100-confidence_level)/2)])

  def sigma_weat_effect_size_precomputed_sims(self, T1, T2, A1, A2, samples, confidence_level):
    effect_sizes = np.zeros(samples)
    for i in range(samples):
        effect_sizes[i] = self.weat_effect_size_precomputed_sims(self._generate_bootstrap(T1), self._generate_bootstrap(T2), self._generate_bootstrap(A1), self._generate_bootstrap((A2)))
    return np.percentile(effect_sizes,[(100-confidence_level)/2,(100-(100-confidence_level)/2)]) 
#    return intervals


  def _random_permutation(self, iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

  def _generate_permutations(self, T1, T2, sample):
    size_of_permutation = min(len(T1), len(T2))
    T1_T2 = T1 + T2
    total_possible_permutations = math.factorial(len(T1_T2)) / math.factorial(size_of_permutation) / math.factorial((len(T1_T2)-size_of_permutation))
    logging.info("Number of possible permutations: %d", total_possible_permutations)
    if not sample or sample >= total_possible_permutations:
      permutations = combinations(T1_T2, size_of_permutation)
    else:
      logging.info("Computing randomly first %d permutations", sample)
      permutations = set()
      while len(permutations) < sample:
        permutations.add(tuple(sorted(self._random_permutation(T1_T2, size_of_permutation))))
    return permutations

  def weat_p_value_precomputed_sims(self, T1, T2, A1, A2, permutations):
    logging.info("Calculating p value ... ")
    T1_T2 = T1 + T2
    observed_test_stats_over_permutations = []
    for Xi in permutations:
      Yi = filterfalse(lambda w: w in Xi, T1_T2)
      observed_test_stats_over_permutations.append(self.differential_association_precomputed_sims(Xi, Yi, A1, A2))
      if len(observed_test_stats_over_permutations) % 100000 == 0:
        logging.info("Iteration %s finished", str(len(observed_test_stats_over_permutations)))
    unperturbed = self.differential_association_precomputed_sims(T1, T2, A1, A2)
    is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
    return is_over.sum() / is_over.size

  def weat_stats_precomputed_sims(self, T1, T2, A1, A2, bootstraps, confidence, sample_p=None):
    test_statistic = self.differential_association_precomputed_sims(T1, T2, A1, A2)
    interval_lowS, interval_highS = self.sigma_differential_association_precomputed_sims(T1, T2, A1, A2, bootstraps, confidence)
    effect_size = self.weat_effect_size_precomputed_sims(T1, T2, A1, A2)
    interval_low, interval_high = self.sigma_weat_effect_size_precomputed_sims(T1, T2, A1, A2, bootstraps, confidence)
    #permutations = self._generate_permutations(T1, T2, sample=sample_p)
    #p = self.weat_p_value_precomputed_sims(T1, T2, A1, A2, permutations)
    p = 0
    return test_statistic, interval_lowS, interval_highS, effect_size, interval_low, interval_high, p

  def _create_vocab(self):
    """
    >>> weat = CAWEAT(None); weat._create_vocab()
    :return: all
    """
    all = []
    for i in range(1, 10):
      t1, t2, a1, a2 = getattr(self, "weat_" + str(i))()
      all = all + t1 + t2 + a1 + a2
    for i in range(1, 2):
      t1, a1, a2 = getattr(self, "wefat_" + str(i))()
      all = all + t1 + a1 + a2
    all = set(all)
    return all

  def run_test_precomputed_sims(self, target_1, target_2, attributes_1, attributes_2, bootstraps=1000, confidence=95, sample_p=None, similarity_type="cosine"):
    """Run the WEAT test for differential association between two
    sets of target words and two sets of attributes.

    RETURNS:
        (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
        e is the effect size, and p is the one-sided p-value measuring the
        (un)likeliness of the null hypothesis (which is that there is no
        difference in association between the two target word sets and
        the attributes).

        If e is large and p small, then differences in the model between
        the attribute word sets match differences between the targets.
    """
    vocab = target_1 + target_2 + attributes_1 + attributes_2
    self._build_vocab_dict(vocab)
    T1 = self.convert_by_vocab(target_1)
    T2 = self.convert_by_vocab(target_2)
    A1 = self.convert_by_vocab(attributes_1)
    A2 = self.convert_by_vocab(attributes_2)
    while len(T1) < len(T2):
      logging.info("Popped T2 %d", T2[-1])
      T2.pop(-1)
    while len(T2) < len(T1):
      logging.info("Popped T1 %d", T1[-1])
      T1.pop(-1)
    while len(A1) < len(A2):
      logging.info("Popped A2 %d", A2[-1])
      A2.pop(-1)
    while len(A2) < len(A1):
      logging.info("Popped A1 %d", A1[-1])
      A1.pop(-1)
    assert len(T1)==len(T2)
    assert len(A1) == len(A2)
    self._build_embedding_matrix()
    self._init_similarities(similarity_type)
    return self.weat_stats_precomputed_sims(T1, T2, A1, A2, bootstraps, confidence, sample_p)

  def weat_1(self, input_file, lang):
    df = self._load_weat_from_file(input_file)
    # if input_file.endswith('.tsv'):
    #   # df = pd.read_csv('./data/CA-WEATv1.tsv', sep='\t',index_col=False)
    #   df = pd.read_csv(input_file, sep='\t',index_col=False)
    # else:
    #   df = pd.read_json(input_file, orient="index", dtype=False) 
    #   df.index.name = 'LANG'
    #   df.reset_index(level=0, inplace=True)
    #   print(df)

      # df = pd.read_csv(input_file, orient="index", dtype=False)
    # print(df.loc[df['LANG'] == lang]['INSECTS'])
    targets_1 = df.loc[df['LANG'] == lang]['INSTRUMENTS'].values[0].replace(', ', ',').split(',')
    targets_2 = df.loc[df['LANG'] == lang]['INSECTS'].values[0].replace(', ', ',').split(',')
    attributes_1 = df.loc[df['LANG'] == lang]['PLEASANT'].values[0].replace(', ', ',').split(',')
    attributes_2 = df.loc[df['LANG'] == lang]['UNPLEASANT'].values[0].replace(', ', ',').split(',')

    return targets_1, targets_2, attributes_1, attributes_2

  def weat_2(self, input_file, lang):
    df = self._load_weat_from_file(input_file)
    # if input_file.endswith('.tsv'):
    #   # df = pd.read_csv('./data/CA-WEATv1.tsv', sep='\t',index_col=False)
    #   df = pd.read_csv(input_file, orient="index", dtype=False)
    # else:
    #   df = pd.read_json(input_file, orient="index", dtype=False) #csv(input_file, sep='\t',index_col=False)

    targets_1 = df.loc[df['LANG'] == lang]['INSTRUMENTS'].values[0].replace(', ', ',').split(',')
    targets_2 = df.loc[df['LANG'] == lang]['WEAPONS'].values[0].replace(', ', ',').split(',')
    attributes_1 = df.loc[df['LANG'] == lang]['PLEASANT'].values[0].replace(', ', ',').split(',')
    attributes_2 = df.loc[df['LANG'] == lang]['UNPLEASANT'].values[0].replace(', ', ',').split(',')

    return targets_1, targets_2, attributes_1, attributes_2

  @staticmethod
  def _load_weat_from_file(input_file):
    if input_file.endswith('.tsv'):
      df = pd.read_csv(input_file, sep='\t',index_col=False)
    else:
      df = pd.read_json(input_file, orient="index", dtype=False) 
      df.index.name = 'LANG'
      df.reset_index(level=0, inplace=True)
      print(df)
    return df

    
def load_embedding_dict(vocab_path="", vector_path="", embeddings_path="", glove=False, postspec=False):
  """
  >>> _load_embedding_dict()
  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """
  if embeddings_path != "":
    embd_dict = utils.load_embeddings(embeddings_path, word2vec=False)
    return embd_dict

def format_output(result):
    max_assoc = '{:.2f}'.format(round(result[2]-result[0],2))
    min_assoc = '{:.2f}'.format(round(result[0]-result[1],2))
    max_eff = '{:.2f}'.format(round(result[5]-result[3],2))
    min_eff = '{:.2f}'.format(round(result[3]-result[4],2))
    formatted = '{:.2f}'.format(round(result[0],2)) + '$^{+' + max_assoc + '}_{-' + min_assoc +'}  ' + \
                '{:.2f}'.format(round(result[3],2)) + '$^{+' + max_eff + '}_{-' + min_eff +'}  ' + \
                '{:.2f}'.format(round(result[6],4))
    return formatted


def main():
  def boolean_string(s):
    if s not in {'False', 'True', 'false', 'true'}:
      raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'
    
  parser = argparse.ArgumentParser(description="Running CA-WEAT")
  # As of now, the default is still v1, in tsv format
  parser.add_argument("--input", type=str, dest="input", required=False, default="./data/CA-WEATv1.tsv",
                 help = "Full/relative path to the input file (tsv or json)")
  parser.add_argument("--test_number", type=int, help="Number of the weat test to run", required=False)
  parser.add_argument("--permutation_number", type=int, default=0,
                      help="Number of permutations (otherwise none will be run)", required=False)
  parser.add_argument("--confidence_level", type=int, default=95, help="Confidence level in %", required=False)
  parser.add_argument("--bootstrap_number", type=int, default=5000, help="Number of bootstrap sets", required=False)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results", required=False)
  parser.add_argument("--lower", type=boolean_string, default=False, help="Whether to lower the vocab" , required=True)
  parser.add_argument("--accents", type=boolean_string, default=False, help="Whether to remove accents/diacritics", required=True)
  parser.add_argument("--phrases", type=boolean_string, default=True, help="Accept multiwords in WEAT lists", required=False)
  parser.add_argument("--similarity_type", type=str, default="cosine", help="Which similarity function to use",
                      required=False)
  parser.add_argument("--embedding_vocab", type=str, help="Vocab of the embeddings")
  parser.add_argument("--embedding_vectors", type=str, help="Vectors of the embeddings")
  parser.add_argument("--is_vec_format", type=boolean_string, default=True, help="Whether embeddings are in vec format")
  parser.add_argument("--embeddings", type=str, help="Vectors and vocab of the embeddings")
  parser.add_argument("--lang", type=str, default="en", help="Language to test")
  args = parser.parse_args()

  start = time.time()
  logging.basicConfig(level=logging.INFO)
  random.seed(3642)

  input_file = args.input


  caweat = CAWEAT()
  if args.test_number == 1:
    targets_1, targets_2, attributes_1, attributes_2 = caweat.weat_1(input_file, args.lang)
    logging.info("CA-WEAT1 started")
  elif args.test_number == 2:
    targets_1, targets_2, attributes_1, attributes_2 = caweat.weat_2(input_file, args.lang)
    logging.info("CA-WEAT2 started")
  else:
    raise ValueError("Only WEAT 1 and 2 are supported")

  if args.lower:
    targets_1 = [t.lower() for t in targets_1]
    targets_2 = [t.lower() for t in targets_2]
    attributes_1 = [a.lower() for a in attributes_1]
    attributes_2 = [a.lower() for a in attributes_2]

  if args.accents:
    targets_1 = [unidecode.unidecode(t) for t in targets_1]
    targets_2 = [unidecode.unidecode(t) for t in targets_2]
    attributes_1 = [unidecode.unidecode(a) for a in attributes_1]
    attributes_2 = [unidecode.unidecode(a) for a in attributes_2]

  if args.phrases:
    targets_1 = [t.lstrip().rstrip().replace(' ','_') for t in targets_1]
    targets_2 = [t.lstrip().rstrip().replace(' ','_') for t in targets_2]
    attributes_1 = [a.lstrip().rstrip().replace(' ','_') for a in attributes_1]
    attributes_2 = [a.lstrip().rstrip().replace(' ','_') for a in attributes_2]

  logging.info(input)
  logging.info(targets_1)
  logging.info(targets_2)
  logging.info(attributes_1)
  logging.info(attributes_2)

  if args.is_vec_format:
    logging.info("Embeddings are in vec format")
    embd_dict = load_embedding_dict(embeddings_path=args.embeddings, glove=False)
  else:
    embd_dict = load_embedding_dict(vocab_path=args.embedding_vocab, vector_path=args.embedding_vectors, glove=False)
  caweat.set_embd_dict(embd_dict)

  logging.info("Embeddings loaded")
  logging.info("Running test")
  result = caweat.run_test_precomputed_sims(targets_1, targets_2, attributes_1, attributes_2, args.bootstrap_number, args.confidence_level, args.permutation_number, args.similarity_type)
  nice_result = format_output(result)
  logging.info(nice_result)
  with codecs.open(args.output_file, "w", "utf8") as f:
    f.write("\n") 
    f.write("Config: ")
    f.write(str(args.test_number) + " and ")
    f.write(str(args.lower) + " and ")
    f.write(str(args.permutation_number) + "\n")
    f.write("Result: test_statistic, effect_size, p")
    f.write(nice_result)
    f.write("\n")
    f.write(str(result))
    f.write("\n")
    end = time.time()
    duration_in_hours = ((end - start) / 60) / 60
    f.write(str(duration_in_hours))
    f.close()

if __name__ == "__main__":
  main()
  
