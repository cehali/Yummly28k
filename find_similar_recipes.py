# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import heapq
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


rec_flavcmp_tfidf = pd.read_pickle('rec_flavcmp_tfidf_updated.pkl')
rec_cos = cosine_similarity(rec_flavcmp_tfidf)
np.save('recipes_cosine_similarity_updated.npy', rec_cos)

'''rec_cos_sim = np.load('recipes_cosine_similarity.npy')
recipes_data = json.load(open('metadata27638/merged_file.json'))

nr = 0
for rd in recipes_data:
    if 'Red Butterhead Lettuce and Arugula Salad with Tangerines and Hard' in rd.get('name'):
        rec_idx = nr
        break
    nr += 1


rec_cos_sim_idx = map(list(rec_cos_sim[rec_idx]).index, heapq.nlargest(11, rec_cos_sim[rec_idx]))

for idx in rec_cos_sim_idx:
    print recipes_data[idx].get('name')'''
