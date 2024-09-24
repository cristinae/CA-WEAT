import argparse
import logging
import numpy as np
import pandas as pd

from collections import Counter, OrderedDict


from cleaning.caweat_inspector import Inspector

logging.basicConfig(level=logging.INFO)

# This code is derived from
# https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
# (function matrix_factorization comes from it)
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter"""
    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T
#
# R = [
#
#     [5, 3, 0, 1],
#
#     [4, 0, 0, 1],
#
#     [1, 1, 0, 5],
#
#     [1, 0, 0, 4],
#
#     [0, 1, 5, 4],
#
#     [2, 1, 3, 0],
#
# ]
#
# R = numpy.array(R)
# # N: num of User
# N = len(R)
# # M: num of Movie
# M = len(R[0])
# # Num of Features
# K = 3
#
# P = numpy.random.rand(N, K)
# Q = numpy.random.rand(M, K)
#
# nP, nQ = matrix_factorization(R, P, Q, K)
#
# nR = numpy.dot(nP, nQ.T)

# print(nR)


class Completer(Inspector):

    #  The score assigned to the top term
    MAX_SCORE = 25
    DUMP_PREFIX = "matrix"

    def __init__(self, language, path_to_tsv):
        super(Completer, self).__init__(language, path_to_tsv)
        logging.info("Input file: %s", path_to_tsv)
        logging.info("Language: %s", language)

    def complete(self, category="fruits"):
        logging.info("Working with category: %s", category.upper())
        elements = self._get_x(category)

        R, people, terms = self._matrixy(elements)
        # print(elements)
        R = np.array(R)

        self._dumper(R, people, terms, ".".join([self.DUMP_PREFIX, category, "orig.tsv"]))
        logging.info("Vectorising")

        # N: num of User
        N = len(R)
        # M: num of Movie
        M = len(R[0])
        # Num of Features (TODO not sure what is this)
        K = 3

        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)

        nP, nQ = matrix_factorization(R, P, Q, K)

        nR = np.dot(nP, nQ.T)
        # print(nR)
        self._dumper(nR, people, terms, ".".join([self.DUMP_PREFIX, category, "comp.tsv"]))

        logging.info("Trying to fill the next gap")

        # TODO consider to do people and terms global
        self._next_element4all(R, nR, people, terms)

        # return nR

    @staticmethod
    def _next_element4all(orig, fill, people, terms):
        # print(orig)
        # converting empties into 1s, existing into 0
        orig = pd.DataFrame(orig, index=people, columns=terms)
        fill = pd.DataFrame(fill, index=people, columns=terms)

        orig = orig * -1
        orig += 1

        # print("=" * 20)
        orig = orig.where(orig > 0, 0)
        # orig.fillna()
        # print(orig)

        # print("=" * 20)
        # print("=" * 20)

        # print(fill)
        # print("=" * 20)

        # masking fill: the ones I have already will become 0, because they are irrelevant
        fill = fill.where(orig == 1, 0)
        # print(fill)

        print(fill.idxmax(1))
        # return fill.idxmax(1)


    @staticmethod
    def _dumper(matty, people, terms, file_name):
        df = pd.DataFrame(matty, index=people, columns=terms)
        # print(matty)
        # print(df)
        logging.info("Matrix saved to %s", file_name)
        df.to_csv(file_name, sep="\t")

    def _matrixy(self, terms):
        """
        Converts from a dictionary of terms to a matrix
        TODO evaluate if I return the index of the ids and terms as well
        :param terms: dict, email:list of terms
        :return: matrix
        """
        emails = []
        email_to_index = {}
        term_to_index = {}

        # get vocabulary
        vocabulary = set()
        for i, key in enumerate(terms):
            emails.append(key)
            email_to_index[key] = i
            vocabulary.update(terms[key])
        vocabulary = list(vocabulary)
        vocabulary.sort()

        for i, key in enumerate(vocabulary):
            term_to_index[key] = i
        logging.debug("Full vocabulary:", vocabulary)

        matty = np.zeros([len(email_to_index), len(vocabulary)])
        logging.info("Zero matrix created")
        # print(matty)
        logging.debug("Shape of the matrix:", matty.shape)
        for email in terms:
            current = set()
            score = self.MAX_SCORE
            for tok in terms[email]:
                if tok not in current and "xx" not in tok :
                    matty[email_to_index[email]][term_to_index[tok]] = score
                    score -= 1
                    current.add(tok)
                else:
                    logging.info("Volunteer %s duplicated %s", email, tok)
            if score > 0:
                logging.warning("%s has at least one missing", email)
        logging.info("Preferences filled into matrix")
        # print(matty)
        return matty, emails, vocabulary

            # AQUI LLAMAR LA MATRIZ Y DEVOLVERLA. TENGO QUE EMPEZAR EN 25, de izquierda a derecha


def main(param):
    weat = Completer(param["language"], param["input"]  )
    weat.complete(param["category"])
    # if param["mode"]:
    #     weat.find_empties()
    # else:
    #     weat.report_duplicates()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # To be decided: whether to run this for all categories
    parser.add_argument('-c', "--category", dest="cat",
                        required=False, default="fruits",
                        help="which category to pseudo-complete")

    parser.add_argument('-l', "--language", dest="lang",
                        required=False, default="Spanish",
                        help = "language")

    parser.add_argument('-t', "--tsv", dest="input",
                        required=True,
                        help="full path to the tsv input file (e.g., ../CulturalAwareWEATrespostes-italians.tsv)")


    parser.set_defaults(mode=False)

    arguments = parser.parse_args()

    param = OrderedDict()
    param["category"] = arguments.cat
    param["language"] = arguments.lang
    param["input"] = arguments.input

    main(param)