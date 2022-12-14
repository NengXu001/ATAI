from transformer import trans_pred
from get_relation import get_relations
from KG_search import test_dis
from KG_search import SPA
import KG_search
import numpy as np
import csv
import numpy as np
import os
import rdflib
import pandas as pd
from sklearn.metrics import pairwise_distances
import embedding as em
import json

"""
Define the graph class, preload the large datasets when the agent is initialized."
"""

class graph_:

    def __init__(self) -> None:

        self.g, self.nodes, self.predicates = KG_search.preprocess_graph('14_graph.nt')
        self.entity_emb, self.relation_emb, self.ent2id, self.id2ent, self.rel2id, self.ent2lbl = em.Emb_preprocess(self.g)
        f = open('images.json')
        self.json_object = json.load(f)

    def return_kg(self):
        return self.g, self.nodes, self.predicates

    def return_em(self):
        return self.entity_emb, self.relation_emb, self.ent2id, self.id2ent, self.rel2id, self.ent2lbl
    
    def return_js(self):
        return self.json_object