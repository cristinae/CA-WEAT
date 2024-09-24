# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import pandas as pd


from collections import Counter, OrderedDict


logging.basicConfig(level=logging.INFO)

# albarron, 24/08/23. This is legacy from the time when no CLI was there. The
# input file should simply come from the command line

# albarron, 10/11/23. In order to create these files, go to the google sheets
# and download a sheet as .tsv
PATH = "/Users/albarron/projects/ca-weat"
TSVS = {
	"Spanish": "CulturalAwareWEAT-es.tsv",
    "Italian":"CulturalAwareWEATrespostes-italians.tsv",
    "German": "CulturalAwareWEATrespostes-alemanys.tsv",
    "Resta": "CulturalAwareWEATrespostes-resta.tsv",
    "WRONG":  "CulturalAwareWEATrespostes-Respostesalformulari5.tsv"
}

# These are the names of the different sheets, downloaded from the google sheets file.
LANG_SPANISH = "Spanish"
LANG_FRENCH = "fr"
LANG_GERMAN = "German"
LANG_GREEK = "el"
LANG_ITALIAN = "Italian"
LANG_RESTA = "Resta"

class Inspector:
    """
    A simplistic class to inspect the CA WEAT records
    """

    categories = ["fruits", "weapons", "flowers", "instruments", "insects", "pleasant", "unpleasant"]
    """
    The 7 categories of the CA WEAT study
    """

    def __init__(self, language, path_to_tsv):
        """

        :param language: str, language we are dealing with
        :param path_to_tsv: str
        """
        self.relevant_columns = self._get_relevant_columns(language)
        logging.info("Loading file %s", (path_to_tsv))
        self.df = pd.read_csv(path_to_tsv,
                              sep="\t",
                              header=0,
                              usecols=self.relevant_columns.values(),
                              index_col=self.relevant_columns.get("email"))

    @staticmethod
    def _get_relevant_columns(language):
        """
        Loads the necessary headers from the tsv to refer to them when loading the pandas dataframe

        :param language: Italian, German, Spanish (if none, defaults)
        :return: dict, links unique names with headers (which vary according to the language)
        The default is a mix of Catalan and English
        """

        if language == "Italian":
            relevant_columns = {
                "email": "adreça electrònica",
                "lang": "Qual è la tua lingua madre? (quella che dovresti usare per compilare questo modulo)",
                "fruits": "prima le cose facili. Scrivi il nome di 25 fRUTTI nella tua lingua. per favore, scrivili tutti in una sola riga e separa ogni frutto con una virgola.",
                "weapons": "passiamo alle armi. Scrivi il nome di 25 aRMI nella tua lingua. per favore, scrivile tutte in una sola riga e separa ogni arma con una virgola.",
                "flowers": "È l'ora dei fiori. Scrivi il nome di 25 fIORI nella tua lingua. per favore, scrivili tutti in una sola riga e separa ogni fiore con una virgola.",
                "instruments": "E la musica? Scrivi il nome di 25 STRUMENTI MUSIcaLI nella tua lingua. per favore, scrivili tutti in una sola riga e separa ogni strumento con una virgola.",
                "insects": "siamo quasi alla fine. insetti! Scrivi il nome di 25 INSETTI nella tua lingua. per favore, scrivili tutti su una sola riga e separa ogni insetto con una virgola.",
                "pleasant": "prima di finire, qualcosa di leggermente diverso. per favore, scrivi 25 concetti che pensi siano GraDEVOLI. può essere un sostantivo o un aggettivo. Come prima, scrivili tutti in una sola riga e separali con una virgola.",
                "unpleasant": "Ultima domanda! Ora scrivi 25 concetti che pensi siano SGraDEVOLI. può essere un sostantivo o un aggettivo. per favore, scrivili tutti in una sola riga e separali con una virgola."
            }

        elif language == "French":
            relevant_columns = {
                "email": "Adreça electrònica",
                "lang": "Quelle est votre langue maternelle ? (celle que vous devez utiliser pour remplir ce formulaire)",
                "fruits": "Commençons par les choses simples. Écrivez le nom de 25 FRUITS dans votre langue. Écrivez-les tous sur une seule ligne et séparez chaque nom de fruit par une virgule.",
                "weapons": "Nous continuons avec les armes. Écrivez le nom de 25 ARMES dans votre langue. Veuillez les écrire tous sur une seule ligne et séparer chaque nom d'arme par une virgule.",
                "flowers": "Et maintenant, les fleurs. Écrivez le nom de 25 FLEURS dans votre langue. Veuillez les écrire sur une seule ligne et séparer chaque nom de fleur par une virgule.",
                "instruments": "Qu'en est-il de la musique ? Écrivez le nom de 25 INSTRUMENTS DE MUSIQUE dans votre langue. Veuillez les écrire tous sur une seule ligne et séparer chaque nom d'instrument par une virgule.",
                "insects": "On approche de la fin. Insectes ! Écrivez le nom de 25 INSECTES dans votre langue. Écrivez-les tous sur une seule ligne et séparez chaque nom d'insecte par une virgule.",
                "pleasant": "Avant de terminer, quelque chose de légèrement différent. Veuillez nommer 25 concepts qui, selon vous, sont des choses PLAISANTES. Il peut s'agir d'un nom ou d'un adjectif, mais pas un nom propre ! Comme précédemment, écrivez-les tous sur une seule ligne et séparez chaque concept par une virgule.",
                "unpleasant": "Dernière question ! Nommez maintenant 25 concepts qui, selon vous, sont DÉSAGRÉABLES. Il peut s'agir d'un nom ou d'un adjectif, mais pas un nom propre. Écrivez-les tous sur une seule ligne et séparez chaque concept par une virgule."
            }
        elif language == "German":
            relevant_columns = {
                "email": "Adreça electrònica",
                "lang": "Which is your native language? (the one you should use to fill this form)",
                "fruits": "Easy things first. Write the name of 25 FRUITS in your language. Please, write all of them in a single line and separate each fruit with a comma.",
                "weapons": "Let's go with weapons now. Write the name of 25 WEAPONS in your language. Please, write all of them in a single line and separate each weapon name with a comma.",
                "flowers": "Time for flowers. Write the name of 25 FLOWERS in your language. Please, write all of them in a single line and separate each flower with a comma.",
                "instruments": "What about music? Write the name of 25 MUSICAL INSTRUMENTS in your language. Please, write all of them in a single line and separate each instrument with a comma.",
                "insects": "Getting close to the end. Insects! Write the name of 25 INSECTS in your language. Please, write all of them in a single line and separate each insect with a comma.",
                "pleasant": "Before finishing, something slightly different. Please, write 25 concepts that you think are PLEASANT things. This can be a noun or an adjective. As before, please, write all of them in a single line and separate each concept with a comma.",
                "unpleasant": "Last question! Please, now write 25 concepts that you think are UNPLEASANT. This can be a noun or an adjective. Write all of them in a single line and separate each concept with a comma."
            }
        elif language == "Spanish":
            relevant_columns = {
                "email": "Adreça electrònica",
                "lang": "¿Cuál es tu lengua materna? (la que debes utilizar para rellenar este formulario)",
                "fruits": "Lo fácil primero. Escribe el nombre de 25 FRUTAS en tu idioma. Por favor, escríbelas todas en una sola línea y separa cada fruta con una coma.",
                "weapons": "Vayamos ahora con armas. Escribe el nombre de 25 ARMAS en tu idioma. Por favor, escríbelas todas en una sola línea y separa cada nombre con una coma.",
                "flowers": "Tiempo de flores. Escribe el nombre de 25 FLORES en tu idioma. Por favor, escríbelos todos en una sola línea y separa cada flor con una coma.",
                "instruments": "¿Música? Escribe el nombre de 25 INSTRUMENTOS MUSICALES en tu idioma. Por favor, escríbelos todos en una sola línea y separa cada instrumento con una coma.",
                "insects": "Nos acercamos al final. ¡Insectos! Escribe el nombre de 25 INSECTOS en tu idioma. Por favor, escríbelos todos en una sola línea y separa cada insecto con una coma.",
                "pleasant": "Antes de terminar, algo ligeramente distinto. Por favor, escribe 25 conceptos que te parezcan cosas PLACENTERAS o AGRADABLES. Pueden ser sustantivos o adjetivos, pero no nombres propios! Como antes, escríbelos todos en una sola línea y separa cada concepto con una coma.",
                "unpleasant": "¡Última pregunta! Escribe ahora 25 conceptos que te parezcan DESAGRADABLES. Pueden ser sustantivos o adjetivos, pero como antes, no uses nombres propios. Por favor, escríbelos todos en una sola línea y separa cada concepto con una coma."
            }
        else:
            relevant_columns = {
                "email": "Adreça electrònica",
                "lang": "Which is your native language? (the one you should use to fill this form)",
                "fruits": "Easy things first. Write the name of 25 FRUITS in your language. Please, write all of them in a single line and separate each fruit with a comma.",
                "weapons": "Let's go with weapons now. Write the name of 25 WEAPONS in your language. Please, write all of them in a single line and separate each weapon name with a comma.",
                "flowers": "Time for flowers. Write the name of 25 FLOWERS in your language. Please, write all of them in a single line and separate each flower with a comma.",
                "instruments": "What about music? Write the name of 25 MUSICAL INSTRUMENTS in your language. Please, write all of them in a single line and separate each instrument with a comma.",
                "insects": "Getting close to the end. Insects! Write the name of 25 INSECTS in your language. Please, write all of them in a single line and separate each insect with a comma.",
                "pleasant": "Before finishing, something slightly different. Please, write 25 concepts that you think are PLEASANT things. This can be a noun or an adjective. As before, please, write all of them in a single line and separate each concept with a comma.",
                "unpleasant": "Last question! Please, now write 25 concepts that you think are UNPLEASANT. This can be a noun or an adjective. Write all of them in a single line and separate each concept with a comma."
            }

        return relevant_columns

    def find_empties(self):
        """
        Displays the records that contain empty entries.
        Those entries must be deleted before running the code to search for
        duplicates

        :return: null
        """
        logging.info("Looking for empty entries")
        # print(self.df.where(pd.isnull(self.df)))
        missing_cols, missing_rows = (
            (self.df.isnull().sum(x) | self.df.eq('').sum(x))
                .loc[lambda x: x.gt(0)].index
            for x in (0, 1)
        )
        print(self.df.loc[missing_rows, missing_cols])


    def report_duplicates(self):
        """
        Looks for entries that contain duplicated items and displays a summary
        :return:
        """
        logging.info("Looking for duplicates")
        for cat in self.categories:
            print(cat.upper())
            elements = self._get_x(cat)

            for i in elements:
                counts = dict(Counter(elements[i]))
                duplicates = {key: value for key, value in counts.items() if value > 1}
                if duplicates:
                    print("Volunteer", i, "has duplicates", duplicates)
                    print("the whole list is", " , ".join(elements[i]))
            print()

    def to_string(self):
        return self.df.to_string

    def get_dataframe(self):
        return self.df

    def get_dataframe_normalizedcolnames(self):
        df = self.df.copy()

        TYPE	WHO	LANG	BORN PLACE	WEAPONS	FLOWERS	INSTRUMENTS	INSECTS	PLEASANT	UNPLEASANT

        return df



    def get_flowers(self):
        return self._get_x("flowers")

    def get_fruits(self):
        return self._get_x("fruits")

    def get_insects(self):
        return self._get_x("insects")

    def get_instruments(self):
        return self._get_x("instruments")

    def get_pleasant(self):
        return self._get_x("pleasant")

    def get_unpleasant(self):
        return self._get_x("unpleasant")

    def get_weapons(self):
        return self._get("weapons")

    def _get_x(self, col):
        x = dict()
        for i, row in self.df.iterrows():
            if row[self.relevant_columns.get(col)] is not None:
                x[i] = [w.strip() for w in row[self.relevant_columns.get(col)].lower().split(",")]
        return x

#
# for i in fruits:
#     print(i)
#     # print(fruits[i])
#     counts = dict(Counter(fruits[i]))
#     duplicates = {key: value for key, value in counts.items() if value > 1}
#     print(duplicates)
#
#     print(len(fruits[i]))
#
#
#     # print(len(set(fruits[i])))

def main(param):
    # weat = Inspector(LANG_GERMAN, os.path.join(PATH, TSVS[LANG_GERMAN]))
    # weat = Inspector(LANG_ITALIAN, os.path.join(PATH, TSVS[LANG_ITALIAN]))
    # weat = Inspector(LANG_RESTA, os.path.join(PATH, TSVS[LANG_RESTA]))
    weat = Inspector(param["language"], param["input"])

    if param["mode"]:
        weat.find_empties()
    else:
        weat.report_duplicates()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', "--language", dest="lang", required=False, default=LANG_ITALIAN,
                        help = "language")

    parser.add_argument('-t', "--tsv", dest="input", required=False, default=os.path.join(PATH, TSVS[LANG_ITALIAN]),
                        help="tsv input file")

    parser.add_argument('-e', "--empties", dest="mode", required=False, action='store_true',
                        help="Operation: search for empties if present, find duplicates if not (default)")

    parser.set_defaults(mode=False)

    arguments = parser.parse_args()

    param = OrderedDict()
    param["language"] = arguments.lang
    param["input"] = arguments.input
    param["mode"] = arguments.mode

    main(param)
