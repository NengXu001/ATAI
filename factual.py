from transformer import trans_pred
from get_relation import get_relations
from KG_search import test_dis
from KG_search import SPA
from KG_search import SPA_image
import numpy as np
import csv
import numpy as np
import os
import rdflib
import pandas as pd
from sklearn.metrics import pairwise_distances
import embedding as em
from preload import graph_
import editdistance
from crowd_sourcing import crowd_sourcing
from crowd_sourcing import js_search
import json
import random

def query_answer(question,g_):

    answer = None

    cro_data = pd.read_csv('crowd_opt.csv')
    
    g, nodes, predicates = g_.return_kg()
    entity_emb, relation_emb, ent2id, id2ent, rel2id, ent2lbl = g_.return_em()

    entities = trans_pred(question)
    entity = ' '.join(entities)
    relation = get_relations(question)

    match_node, match_pred = test_dis(nodes,predicates,entity,relation)

    result_kg, final_node, final_pred = SPA(g, match_node, match_pred)

    em_entity = rdflib.term.URIRef(match_node[0][0])

    while True:
        try:
            em_predicate = rdflib.term.URIRef(match_pred[0][0])
            result_em = em.query(em_entity, em_predicate, entity_emb, relation_emb, ent2id, id2ent, rel2id, ent2lbl)
            break

        except KeyError or TypeError:

            if final_pred == None:

                while True:

                    try:
                        answer,support,reject,interatio = crowd_sourcing(cro_data, match_node[0][0], ent2lbl)
                        break
                    except IndexError: return "Sorry I'm just a robot X). Could you ask in another way maybe?"

                return "The {} is {}  --according to the crowd sourcing, who had an inter-rater agreement of {} in this batch. The answer distribution for this specific task was {} support votes and {} reject vote.\n ".format(match_pred[0][1],answer,interatio,support,reject)

            answer_kg = "Hey, According to Knowledge Graph, I found the {} of {} is {}. \n".format(final_pred, final_node, result_kg)
            return answer_kg

    if result_kg: 
        answer_kg = "Hey, based on the Knowledge Graph, I found the {} of {} is {}.\n ".format(final_pred, final_node, result_kg)
        answer,support,reject,interatio = crowd_sourcing(cro_data, match_node[0][0], ent2lbl)
        if answer != 'None':
            answer_kg = answer_kg +  "[Crowd, inter-rater agreement: {} .The answer distribution for this specific task was {} support votes and {} reject votes].\n".format(interatio,support,reject)
        
        answer_em = 'And the answer suggested by embeddings: {}.'.format(result_em)

        answer = answer_kg + answer_em
        
    else: answer = 'Hey, unfortunately, no result found by Knowledge graph, but according to embedding, the answer could be {}'.format(result_em)

    if answer == None: 
        answer = "Sorry I'm just a robot X) "
    
    return answer

def recomendation(question,g_):

     entity_emb, _, ent2id, id2ent, _, ent2lbl = g_.return_em()

     recom_list = []
     recom = []
     g, nodes, predicates = g_.return_kg()

     entities = trans_pred(question)

     for entity in entities:
        relation = 'genre'
        match_node,_ = test_dis(nodes,predicates,entity,relation)

        for en in match_node:
            em_entity = rdflib.term.URIRef(en[0])
            recom_list.extend(em.reco(em_entity, entity_emb, ent2id, id2ent, ent2lbl))

        for i in recom_list:
            dist = [(editdistance.eval(entity, i)>=3) for entity in entities]
            if recom_list.count(i)>=2 and all(dist):
                recom.append(i)
        
        for entity in entities:
            for i in recom:
                if entity in i: recom.remove(i)
        
     recom = list(set(recom[:3]))

     while True:
        try:
            if len(recom) == 1: recom = recom[0]
            elif len(recom) == 2: recom =  " and ".join(recom)
            else: recom =  "{};{} and {}".format(recom[0],recom[1],recom[2])
            break

        except IndexError:
            return "My dream is to become an omniscient robot, but still a lot to learn."


     answer = ["Based on what I know, it's a good option for you to try {}".format(recom),
                "Try {}, my freind".format(recom),
                "If you trust me, please try {}".format(recom)]

     i = random.randint(0, 2)

     return answer[i]


def show_pic(question,g_):

    show_list = []

    g, nodes, predicates = g_.return_kg()
    json_object = g_.return_js()

    entities = trans_pred(question)
    relation = 'IMDb ID'

    for entity in entities:

        match_node, match_pred = test_dis(nodes,predicates,entity,relation)

        result_kg,_,_ = SPA_image(g, match_node, match_pred)

        while True:
            try:
                image = js_search(json_object, result_kg)
                break
            except IndexError:
                return "Sorry I'm just a robot X). Could you ask in another way maybe?"

        image = 'image:' + image.replace('.jpg','') +'\n'
        show_list.append(image)

    return 'I hope this is what you want:)\n' + ''.join(show_list)

def main(question,g_):

    Reco_word = ['Recommend','recommend','recommendation','recommendations','similar']
    pic_word = ['image','picture','Picture','look','looks','photo','see']


    for i in Reco_word:
        if i in question: return recomendation(question,g_)

    for j in pic_word:
        if j in question: return show_pic(question,g_)
    
    return query_answer(question,g_)


# text = ["Who is the director of Star Wars: Episode VI - Return of the Jedi?",
#         "Who is the screenwriter of The Masked Gang: Cyprus?"]

# text = ['Who is the director of Good Will Hunting?',
#         'Who directed The Bridge on the River Kwai?',
#         "Who is the director of Star Wars: Episode VI - Return of the Jedi?",
#         "What is the genre of Good Neighbors?",
#         "Who is the screenwriter of The Masked Gang: Cyprus?",
#         "What is the MPAA film rating of Weathering with You?"]

# text_recom = ['Recommend movies similar to Hamlet and Othello.',
#      'Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?',
#      'Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween.']

# text_pic = ['Show me a picture of Halle Berry.',
#             'What does Julia Roberts look like?',
#             'Let me know what Sandra Bullock looks like.'
# ]

# for te in text:
#     print(main(te,g_))

# for te in text_recom:
#     print(recomendation(te,g_))

# for te in text_pic:
#     print(show_pic(te,g_))