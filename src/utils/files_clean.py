from natsort import natsorted
import unicodedata
import pandas as pd

def natural_string_sort(list2sort):
    return natsorted(list2sort, key=lambda y: y.lower())

# Definimos la función de CORRECCIÓN de TILDES y Ñs
def strip_accents(s):
    """
    Remove accents,ñ and other symbols
            Args:
    			s (str): String to be stripped
    		Return:
    			string withouth punctuation symbols and others
    """
    return ''.join(c for c in unicodedata.normalize('NFD', (s))if unicodedata.category(c) != 'Mn')

def save_matrixes(output_path, matrix2save, id_column=[], rename_cols=[]):
    df = pd.DataFrame(matrix2save)
    if(id_column!=[]):
        df["ids"] = id_column
    if(rename_cols!=[]):
        for i in range(0,len(rename_cols)):
            df.columns.values[i] = rename_cols[i]
    df.to_csv(output_path, index=False, sep=";")
    return df