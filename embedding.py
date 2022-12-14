# imports
import csv
import numpy as np
import os
import rdflib
import pandas as pd
from sklearn.metrics import pairwise_distances
from KG_search import test_dis
from KG_search import SPA

def Emb_preprocess(g):

    WD = rdflib.Namespace('http://www.wikidata.org/entity/')
    WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
    DDIS = rdflib.Namespace('http://ddis.ch/atai/')
    RDFS = rdflib.namespace.RDFS
    SCHEMA = rdflib.Namespace('http://schema.org/')

    # load the embeddings
    entity_emb = np.load(os.path.join('data', 'entity_embeds.npy'))
    relation_emb = np.load(os.path.join('data', 'relation_embeds.npy'))
    entity_file = os.path.join('data', 'entity_ids.del')
    relation_file = os.path.join('data', 'relation_ids.del')

    # load the dictionaries
    with open(entity_file, 'r') as ifile:
        ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
        id2ent = {v: k for k, v in ent2id.items()}
    with open(relation_file, 'r') as ifile:
        rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
        id2rel = {v: k for k, v in rel2id.items()}

    ent2lbl = {ent: str(lbl) for ent, lbl in g.subject_objects(RDFS.label)}

    return entity_emb, relation_emb, ent2id, id2ent, rel2id, ent2lbl

def query(entity, predicate, entity_emb, relation_emb, ent2id, id2ent, rel2id, ent2lbl):
    # entity = WD['Q59692464']

    movie_emb = entity_emb[ent2id[entity]]

    # Find the predicate (relation) of the genre (https://www.wikidata.org/wiki/Property:P136 is the ID for "genre")
    # genre = WDT['P1657']
    genre_emb = relation_emb[rel2id[predicate]]

    # combine according to the TransE scoring function
    lhs = movie_emb + genre_emb

    # compute distance to *any* entity
    distances = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)

    # find most plausible tails
    most_likely = distances.argsort()

    answer = []

    # show most likely entities
    for rank, idx in enumerate(most_likely[:20][:3]):
        rank = rank + 1
        ent = id2ent[idx] # eg: https://www.wikidata.org/wiki/Q157443
        lbl = ent2lbl[ent] # eg: 'comedy film'
        answer.append(lbl)

    if len(answer) == 1: return answer[0]
    elif len(answer) == 2: return " and ".join(answer)
    else: return "{}; {} and {}".format(answer[0],answer[1],answer[2])

    # return ";".join(answer)


def reco(em_entity, entity_emb, ent2id, id2ent, ent2lbl):
    # Find the Wikidata ID for the movie 

    WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
    while True:
        try:
            movie_id = ent2id[em_entity]
            break

        except KeyError or TypeError:
            return []


    # we compare the embedding of the query entity to all other entity embeddings
    distances = pairwise_distances(entity_emb[movie_id].reshape(1, -1), entity_emb, metric='cosine').flatten()

    # and sort them by distance
    most_likely = np.argsort(distances)

    recom = []

    # we print rank, entity ID, entity label, and distance
    for rank, idx in enumerate(most_likely[:25]):
        rank = rank + 1
        ent =  id2ent[idx] # eg: http://www.wikidata.org/entity/Q132863 

        while True:
            try:
                lbl = ent2lbl[ent] 
                break

            except KeyError or TypeError:
                break
        recom.append(lbl)

    return recom

