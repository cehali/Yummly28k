import numpy as np
import pandas as pd

recipes_compunds_id = np.load('recipes_compunds_id_updated.npy')

ingr_info = pd.read_csv("ingr_comp/ingr_info.tsv", sep="\t", index_col='# id')

rec_flavcmp = pd.DataFrame(0, index=np.arange(len(recipes_compunds_id)), columns=np.arange(len(ingr_info)))
rec_flavcmp1 = pd.DataFrame(0, index=np.arange(len(recipes_compunds_id)), columns=np.arange(len(ingr_info)))

for idx, rec_cmp in enumerate(recipes_compunds_id):
    for rc in rec_cmp:
        rec_flavcmp.ix[idx, rc] += 1

rec_flavcmp.to_pickle('recipes_flavorcompounds_table_updated.pkl')


#rec_flavcmp = pd.read_pickle('recipes_flavorcompounds_table.pkl')


def make_tfidf(arr):
    arr2 = arr.copy()
    N=arr2.shape[0]
    l2_rows = np.sqrt(np.sum(arr2**2, axis=1)).reshape(N, 1)
    l2_rows[l2_rows==0]=1
    arr2_norm = arr2/l2_rows

    arr2_freq = np.sum(arr2_norm>0, axis=0)
    arr2_idf = np.log(float(N+1) / (1.0 + arr2_freq)) + 1.0

    from sklearn.preprocessing import normalize
    tfidf = np.multiply(arr2_norm, arr2_idf)
    tfidf = normalize(tfidf, norm='l2', axis=1)
    print tfidf.shape
    return tfidf


rec_flavcmp_tfidf = make_tfidf(rec_flavcmp.values)

pd.DataFrame(rec_flavcmp_tfidf).to_pickle('rec_flavcmp_tfidf_updated.pkl')