import numpy as np
import json

data = json.load(open('metadata27638/merged_file.json'))

ingredients_names = np.load('recipes_ingr_mapped_name.npy')

temp1 = data[26999].get('ingredientLines')
temp2 = ingredients_names[26999]

print 'Real ingredients:'
for x in temp1: print x
print 'Ingredients mapped- Flavor Network:'
for y in temp2: print y