# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import heapq
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import numpy as np
import itertools
import csv


def split_ingr(x):
    wnl=WordNetLemmatizer()
    cleanlist=[' '.join(wnl.lemmatize(word.lower()) for word in word_tokenize(re.sub('[^a-zA-Z]',' ',x)))]
    return cleanlist


def map_ingr(rec):
    recipe = []
    recipes_ingr_name = []
    for ingr in rec:
        ingr = ingr.lower()
        ingredient = ingr.replace('[', '').replace(']', '').replace(" u'", '').replace(" '", '')\
            .replace("u'", '').replace('uequipment: ', '').replace("'", '').replace('(', '').replace(')', '')\
            .replace('*', '').replace('\\xa', '').replace('\\xe', '').replace('virgin', '').replace('whole ', '')\
            .replace('available', 'None').replace('unsalted ', '').replace('1-pint', '').replace('basket', '').\
            replace('slicer', '').replace('nonstick vegetable oil spray', 'vegetable oil').\
            replace('salt pepper', 'salt, pepper').replace('strawberries', 'strawberry').replace('(packed)', '')\
            .replace('finely', '').replace('grated', '').replace('available', '').replace('at some butcher', '')\
            .replace('Accompaniment', '').replace('dry white wine', 'white wine').replace('sugar', 'None')\
            .replace('pickle', 'None').replace('water', 'None').replace('bourbon', 'bourbon_whiskey')\
            .replace('all purpose ', '').replace('strawberries', 'strawberry').replace('spinach', 'dried_spinach')\
            .replace('vegetable stock', 'None').replace('vegetable stock broth', 'None')\
            .replace('oriental sesame oil', 'sesame oil').replace('black peppercorns', 'black pepper').replace('whole', '')\
            .replace('all-purpose ', '').replace('asian sesame oil', 'sesame oil').replace('orange slice', 'orange') \
            .replace('cabernet sauvignon', 'cabernet sauvignon wine').replace('spray', '').replace('\bmilk\b', 'skim milk') \
            .replace('tomatoes', 'tomato').replace('pinto beans', 'pinto bean').replace('sausage', 'pork sausage') \
            .replace('raspberries', 'raspberry').replace('nonstick', '') \
            .replace('slice', '').replace('garlic cloves', 'garlic').replace('paprika', 'bell pepper') \
            .replace('garlic clove', 'garlic').replace('salt and black pepper', 'salt, black pepper') \
            .replace('salt and pepper', 'salt, black pepper').replace('blackberries', 'blackberry').replace('salt', 'None') \
            .replace('poblano pepper', 'green bell pepper').replace('toast', 'bread').replace('blackpepper', 'black pepper') \
            .replace('mangoes', 'mango').replace('peaches', 'peach').replace('pomegranate', 'None').replace('+', '') \
            .replace('mahi-mahi fillets', 'dolphin').replace('sweet potatoes', 'sweet potato').replace('chile', 'tabasco pepper')\
            .replace('chili', 'tabasco pepper').replace('chilies', 'tabasco pepper').replace('chilli', 'tabasco pepper')\
            .replace('sriracha', 'tabasco pepper').replace('flour', 'whole grain wheat flour').replace('tumeric', 'turmeric')\
            .replace('yoghurt', 'yogurt').replace('rib', 'beef').replace('chuck', 'beef').replace('mozzarella', 'mozzarella cheese') \
            .replace('sirloin', 'beef').replace('steak', 'beef').replace('fillete', 'raw fish')\
            .replace('spinach', 'dried spinach').replace('curry', 'coriander, turmeric, cumin, cayenne')\
            .replace('low fat', '').replace('reduced fat', '').replace('fatfree', '').replace('nonfat', '')\
            .replace('gluten free', '').replace('free range', '').replace('reduced sodium', '').replace('salt free', '')\
            .replace('sodium free', '').replace('low sodium', '').replace('sweetened', '').replace('unsweetened', '')\
            .replace('large', '').replace('extra large', '').replace('oz ', '').replace('ounces ', '').replace('cups ', '')\
            .replace('cups ', '').replace('tablespoons', '').replace('teaspoons', '').replace('cans ', '')\
            .replace('optional ', '').replace('toppings', '').replace('ounce ', '').replace('cup ', '')\
            .replace('cup ', '').replace('tablespoon', '').replace('teaspoon', '').replace('can ', '')\
            .replace('diced', '').replace('seeded', '').replace('and ', '').replace('ground pepper', 'black pepper')\
            .replace('green sweet pepper', 'green bell pepper').replace('green pepper ', 'green bell pepper')\
            .replace('stock', 'broth').replace('duck', 'turkey').replace('cherry tomato', 'tomato')\
            .replace('chopped', '').replace('stalks', '').replace('stalk', '').replace('lemon grass', 'lemongrass')\
            .replace('mahi mahi fillets', 'dolphin').replace('sirloins', 'beef').replace('steaks', 'beef')\
            .replace('filletes', 'raw fish').replace('ribs', 'beef').replace('Hühnerbrustfilet', 'chicken')\
            .replace('brats', 'pork sausage').replace('egg white', 'egg').replace('worth', '').replace('left to age', '')\
            .replace('chorizo', 'pork sausage').replace('halibut', 'fish').replace('tilapia', 'fish')\
            .replace('pancetta', 'ham').replace('anchovy', 'fish').replace('mahi mahi', 'dolphin').replace('prosciutto', 'ham')\
            .replace('fryer', '').replace('leg of', '').replace('leg', '').replace('deli ', '').replace(' crème fraîche', 'cream')\
            .replace('flounder', 'fish').replace('trout', 'fish').replace('guanciale', 'pork').replace('tilapia', 'fish')\
            .replace('red snapper', 'fish')

        splited_ingredient = split_ingr(ingredient)
        recipe.append(splited_ingredient)

    for rec_ingr in recipe:
        rec_ingr = ' '.join(rec_ingr)
        rec_ingr = rec_ingr.replace(' ', '_')
        cand = []
        if rec_ingr is not None:
            if not ingr_info[ingr_info['ingredient name'] == (rec_ingr)].empty:
                recipes_ingr_name.append(rec_ingr)
            else:
                rec_ingr = rec_ingr.replace('_', ' ')
                for ii in ingr_info['ingredient name'].tolist():
                    ii = ii.replace('_', ' ')
                    if ii in rec_ingr:
                        cand.append(ii.replace(' ', '_'))
                if (len(cand) < 2) and (len(cand) > 0):
                    recipes_ingr_name.append(cand[0])
                if (len(cand) >= 2) and (len(cand) < 5):
                    recipes_ingr_name.append(max(cand, key=len))
                if len(cand) >= 5:
                    temp = (heapq.nlargest(2, cand, key=len))
                    [recipes_ingr_name.append(x) for x in temp]

    return recipes_ingr_name

recipes_data = json.load(open('recipes.json'))

list_recipes = []
for n in range(1, 10):
    rcp = {}
    rcp['meta0000'+str(n)] = recipes_data.get('meta0000'+str(n))
    list_recipes.append(rcp)

for n in range(10, 100):
    rcp = {}
    rcp['meta000'+str(n)] = recipes_data.get('meta000'+str(n))
    list_recipes.append(rcp)

for n in range(100, 1000):
    rcp = {}
    rcp['meta00'+str(n)] = recipes_data.get('meta00'+str(n))
    list_recipes.append(rcp)

for n in range(1000, 10000):
    rcp = {}
    rcp['meta0'+str(n)] = recipes_data.get('meta0'+str(n))
    list_recipes.append(rcp)

for n in range(10000, 27639):
    rcp = {}
    rcp['meta'+str(n)] = recipes_data.get('meta'+str(n))
    list_recipes.append(rcp)


ingr_info = pd.read_csv("ingr_comp/ingr_info.tsv", sep="\t", index_col='# id')
comp_info = pd.read_csv("ingr_comp/comp_info.tsv", sep="\t", index_col='# id')
ingr_comp = pd.read_csv("ingr_comp/ingr_comp.tsv", sep="\t")

recipes_ingr_mapped_name = []

idx = 0
for d in list_recipes:
    idx += 1
    if (idx > 0 and idx < 10):
        id = 'meta0000' + str(idx)
    if (idx >= 10 and idx < 100):
        id = 'meta000' + str(idx)
    if (idx >= 100 and idx < 1000):
        id = 'meta00' + str(idx)
    if (idx >= 1000 and idx < 10000):
        id = 'meta0' + str(idx)
    if (idx >= 10000 and idx < 27639):
        id = 'meta' + str(idx)
    recipes_ingr_mapped_name.append(map_ingr(d[id].get('ingredientLines')))

recipes_ingr_mapped_name = np.array(recipes_ingr_mapped_name)
np.save('recipes_ingr_mapped_name_updated', recipes_ingr_mapped_name)


'''for d in data:
    temp = []
    temp.append('Recipe_nr_' + str(rec_nr))
    [temp.append(x) for x in map_ingr(d.get('ingredientLines'))]
    recipes_ingr_mapped_name.append(temp)
    rec_nr += 1

with open('recipes_ingr_mapped_name.csv', 'wb') as file:
    wr = csv.writer(file)
    for rows1 in recipes_ingr_mapped_name:
        for row1 in rows1:
            wr.writerow(row1)'''