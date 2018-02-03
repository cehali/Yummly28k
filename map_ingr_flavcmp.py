import csv
from codecs import open
import numpy as np
import pandas as pd

ingredients_single_recipe = []
ingrs_id = []
ingredients_id = []
comp_id = []
compounds_id = []

ingredients_names = np.load('recipes_ingr_mapped_name_updated.npy')

ingr_info = pd.read_csv("ingr_comp/ingr_info.tsv", sep="\t", index_col='# id')

ingr_comp = pd.read_csv("ingr_comp/ingr_comp.tsv", sep="\t")

for ingr_names in ingredients_names:
    ingrs_id = []
    for ingredient in ingr_names:
        ingrs_index = ingr_info[ingr_info['ingredient name'] == ingredient].index.tolist()
        if ingrs_index:
            ingrs_id.append(ingrs_index[0])
    ingredients_id.append(ingrs_id)

for rec in ingredients_id:
    comp_id = []
    for ing in rec:
        comp_id.append(ingr_comp['compound id'][ingr_comp['ingredient id'] == ing].tolist())
    compounds_id.append(comp_id)

compounds_id = np.array(compounds_id)
ingredients_id = np.array(ingredients_id)

np.save('recipes_compunds_id_updated', compounds_id)
np.save('ingredients_id_updated', ingredients_id)