import heapq
import pandas as pd
import numpy as np
import json
import itertools
from sklearn.metrics.pairwise import cosine_similarity

recipes_with_everything = []
recipes_without_meats = []
recipes_without_fishes = []
recipes_without_dairy = []
recipes_for_vegetarians = []
recipes_for_vegans = []

recipes_flvcmp_with_everything = []
recipes_flvcmp_without_meats = []
recipes_flvcmp_without_fishes = []
recipes_flvcmp_without_dairy = []
recipes_flvcmp_for_vegetarians = []
recipes_flvcmp_for_vegans = []

recipes_with_everything_tfidf = []
recipes_without_meats_tfidf = []
recipes_without_fishes_tfidf = []
recipes_without_dairy_tfidf = []
recipes_for_vegetarians_tfidf = []
recipes_for_vegans_tfidf = []

rec_cos_sim = np.load('recipes_cosine_similarity.npy')
recipes_data = json.load(open('metadata27638/merged_file.json'))
recipes_compunds_id = np.load('recipes_compunds_id.npy')
recipes_ingr_mapped_name = np.load('recipes_ingr_mapped_name.npy')
rec_flavcmp_tfidf = pd.read_pickle('rec_flavcmp_tfidf.pkl')

recipes_compunds_id_fixed = []
for rec_cmp in recipes_compunds_id:
    temp = []
    for rc in rec_cmp:
        for flv_cmp in rc:
            temp.append(flv_cmp)
    recipes_compunds_id_fixed.append(temp)

meats = ["bacon", "smoked_sausage", "smoked_summer_sausage", "meat", "smoked_pork", "raw_pork", "ham", "beef",
         "roasted_beef", "guineafowl", "fried_chicken", "grilled_beef", "red_meat", "fried_pork", "pork", "frankfurter",
         "raw_chicken", "boiled_chicken", "beef_broth", "uncured_boiled_pork", "boiled_meat", "uncured_smoked_pork",
         "seal", "mutton", "roasted_meat", "turkey", "boiled_beef", "veal", "chicken_liver", "raw_lamb",
         "roasted_turkey", "pickled_ham", "roasted_lamb", "raw_beef", "mutton_liver", "chicken_broth",
         "smoked_pork_belly", "fried_beef", "liver", "ewe", "fried_cured_pork", "roasted_pork", "boiled_pork",
         "beef_tallow", "beef_liver", "cured_ham", "lamb_liver", "lamb", "pork_sausage", "grilled_pork", "uncured_pork",
         "chicken", "raw_turkey", "cured_pork", "pork_liver", "roasted_chicken", "boiled_mutton", "nezara_viridula",
         "musk", "honey", "castoreum", "cod_liver_oil", "chrysocoris_stolli", "gelatin", "auto_oxidized_salmon_oil",
         "fish_oil", "egg", "cockroach", "silk_fibrooin", "feces", "lard", "animal", "bone_oil", "silkworm_chrysalis",
         "oxidized_lard"]

fishes = ["mackerel", "keta_salmon", "dolphin", "trassi", "prawn", "salmon_caviar", "shellfish", "whitefish", "cod",
          "sweetfish", "tuna", "boiled_crab", "eel", "octopus", "mantis_shrimp", "raw_lean_", "fish", "catfish",
          "salmon_roe", "clam", "lobster", "smoked_herring", "shrimp", "globefish", "japanese_seafood", "herring",
          "lean_fish", "krill", "katsuobushi", "smoked_fish", "fish", "haddock", "salmon", "oyster", "sperm_whale_oil",
          "scallop", "fermented_shrimp", "smoked_fatty_fish", "crayfish", "roasted_shrimp", "pilchard",
          "horse_mackerel", "smoked_salmon", "cuttlefish", "mussel", "crab", "raw_fish", "fatty_fish", "salmon_oil",
          "seaweed", "sea_bass", "squid", "whale", "raw_fatty_fish", "pike", "caviar", "sturgeon_caviar"]

dairy = ["cottage_cheese", "cheddar_cheese", "tilsit_cheese", "goat_cheese", "milk_fat", "sour_milk", "sheep_cheese",
         "goat_milk", "yogurt", "sheep_milk", "butter", "milk", "camembert_cheese", "oxidized_milk", "butterfat",
         "limburger_cheese", "cheese", "domiati_cheese", "comte_cheese", "swiss_cheese", "cream", "butter_oil",
         "parmesan", "dairy", "roquefort_cheese", "skim_milk", "gruyere_cheese", "blue_cheese", "provolone_cheese",
         "buttermilk", "oxidized_skim_milk", "cream_cheese", "emmental_cheese", "russian_cheese", "mozzarella_cheese",
         "romano_cheese", "parmesan_cheese", "feta_cheese", "munster_cheese"]

meats_set = set(meats)
fishes_set = set(fishes)
dairy_set = set(dairy)

'''rec_cos_sim_cummulated = []
for rcs in rec_cos_sim:
    rec_cos_sim_cummulated.append(sum(rcs)/len(rcs))'''

index = 0
for rec, rec_ingr, rec_flvcmp in itertools.izip(recipes_data, recipes_ingr_mapped_name, recipes_compunds_id_fixed):

    sim = rec_flavcmp_tfidf.iloc[index]
    if rec_flvcmp:
        recipes_with_everything.append(rec)
        recipes_flvcmp_with_everything.append(rec_flvcmp)
        recipes_with_everything_tfidf.append(sim)

        temp2 = set(rec_ingr)
        meat_inter = temp2.intersection(meats_set)
        fish_inter = temp2.intersection(fishes_set)
        dairy_inter = temp2.intersection(dairy_set)

        if not meat_inter:
            recipes_without_meats.append(rec)
            recipes_flvcmp_without_meats.append(rec_flvcmp)
            recipes_without_meats_tfidf.append(sim)
            if not fish_inter:
                recipes_for_vegetarians.append(rec)
                recipes_flvcmp_for_vegetarians.append(rec_flvcmp)
                recipes_for_vegetarians_tfidf.append(sim)
                if not dairy_inter:
                    recipes_for_vegans.append(rec)
                    recipes_flvcmp_for_vegans.append(rec_flvcmp)
                    recipes_for_vegans_tfidf.append(sim)
        if not fish_inter:
            recipes_without_fishes.append(rec)
            recipes_flvcmp_without_fishes.append(rec_flvcmp)
            recipes_without_fishes_tfidf.append(sim)
        if not dairy_inter:
            recipes_without_dairy.append(rec)
            recipes_flvcmp_without_dairy.append(rec_flvcmp)
            recipes_without_dairy_tfidf.append(sim)
    index += 1


recipes_with_everything_largest = \
    list(set(map(recipes_flvcmp_with_everything.index, heapq.nlargest(1000, recipes_flvcmp_with_everything, key=len))))
recipes_without_meats_largest = \
    list(set(map(recipes_flvcmp_without_meats.index, heapq.nlargest(1000, recipes_flvcmp_without_meats, key=len))))
recipes_without_fishes_largest = \
    list(set(map(recipes_flvcmp_without_fishes.index, heapq.nlargest(1000, recipes_flvcmp_without_fishes, key=len))))
recipes_without_dairy_largest = \
    list(set(map(recipes_flvcmp_without_dairy.index, heapq.nlargest(1000, recipes_flvcmp_without_dairy, key=len))))
recipes_for_vegetarians_largest = \
    list(set(map(recipes_flvcmp_for_vegetarians.index, heapq.nlargest(1000, recipes_flvcmp_for_vegetarians, key=len))))
recipes_for_vegans_largest = \
    list(set(map(recipes_flvcmp_for_vegans.index, heapq.nlargest(1000, recipes_flvcmp_for_vegans, key=len))))

'''recipes_with_everything_sim = [recipes_with_everything_sim_cum[x1] for x1 in recipes_with_everything_largest]
recipes_without_meats_sim = [recipes_without_meats_sim_cum[x2] for x2 in recipes_without_meats_largest]
recipes_without_fishes_sim = [recipes_without_fishes_sim_cum[x3] for x3 in recipes_without_fishes_largest]
recipes_without_dairy_sim = [recipes_without_dairy_sim_cum[x4] for x4 in recipes_without_dairy_largest]
recipes_for_vegetarians_sim = [recipes_for_vegetarians_sim_cum[x5] for x5 in recipes_for_vegetarians_largest]
recipes_for_vegans_sim = [recipes_for_vegans_sim_cum[x6] for x6 in recipes_for_vegans_largest]

recipes_with_everything_similarities = set(map(recipes_with_everything_sim.index, heapq.nsmallest(30, recipes_with_everything_sim)))
recipes_without_meats_similarities = set(map(recipes_without_meats_sim.index, heapq.nsmallest(30, recipes_without_meats_sim,)))
recipes_without_fishes_similarities = set(map(recipes_without_fishes_sim.index, heapq.nsmallest(30, recipes_without_fishes_sim)))
recipes_without_dairy_similarities = set(map(recipes_without_dairy_sim.index, heapq.nsmallest(30, recipes_without_dairy_sim)))
recipes_for_vegetarians_similarities = set(map(recipes_for_vegetarians_sim.index, heapq.nsmallest(30, recipes_for_vegetarians_sim)))
recipes_for_vegans_similarities = set(map(recipes_for_vegans_sim.index, heapq.nsmallest(30, recipes_for_vegans_sim)))

recipes_with_everything = [recipes_data[x7].get('name') for x7 in recipes_with_everything_similarities]
recipes_without_meats = [recipes_without_meats[x7].get('name') for x7 in recipes_without_meats_similarities]
recipes_without_fishes = [recipes_without_fishes[x8].get('name') for x8 in recipes_without_fishes_similarities]
recipes_without_dairy = [recipes_without_dairy[x9].get('name') for x9 in recipes_without_dairy_similarities]
recipes_for_vegetarians = [recipes_for_vegetarians[x10].get('name') for x10 in recipes_for_vegetarians_similarities]
recipes_for_vegans = [recipes_for_vegans[x11].get('name') for x11 in recipes_for_vegans_similarities]'''

rec_flavcmp_tfidf_with_everything = [recipes_with_everything_tfidf[x1] for x1 in recipes_with_everything_largest]
rec_flavcmp_tfidf_without_meats = [recipes_without_meats_tfidf[x2] for x2 in recipes_without_meats_largest]
rec_flavcmp_tfidf_without_fishes = [recipes_without_fishes_tfidf[x3] for x3 in recipes_without_fishes_largest]
rec_flavcmp_tfidf_without_dairy = [recipes_without_dairy_tfidf[x4] for x4 in recipes_without_dairy_largest]
rec_flavcmp_tfidf_for_vegetarians = [recipes_for_vegetarians_tfidf[x5] for x5 in recipes_for_vegetarians_largest]
rec_flavcmp_tfidf_for_vegans = [recipes_for_vegans_tfidf[x6] for x6 in recipes_for_vegans_largest]


rec_cos_sim_with_everything = cosine_similarity(rec_flavcmp_tfidf_with_everything)
rec_cos_sim_with_without_meats = cosine_similarity(rec_flavcmp_tfidf_without_meats)
rec_cos_sim_with_without_fishes = cosine_similarity(rec_flavcmp_tfidf_without_fishes)
rec_cos_sim_with_without_dairy = cosine_similarity(rec_flavcmp_tfidf_without_dairy)
rec_cos_sim_with_for_vegetarians = cosine_similarity(rec_flavcmp_tfidf_for_vegetarians)
rec_cos_sim_with_for_vegans = cosine_similarity(rec_flavcmp_tfidf_for_vegans)


def get_recipes_survey(rec_cos_sim, recipes_for, nr_of_rec):
    rec_cos_sim_cummulated = []
    for rcs in rec_cos_sim:
        rec_cos_sim_cummulated.append(sum(rcs)/len(rcs))

    recipes_similarities = \
        set(map(rec_cos_sim_cummulated.index, heapq.nlargest(nr_of_rec, rec_cos_sim_cummulated)))
    recipes = [recipes_for[x] for x in recipes_similarities]
    return recipes


recipes = get_recipes_survey(rec_cos_sim_with_for_vegetarians, recipes_for_vegetarians, 100)

nr = 0
for x in recipes:
    print 'Recipe nr: ' + str(nr) + ' ' + x.get('name')
    print x.get('attribution').get('url')
    print x.get('images')
    nr += 1