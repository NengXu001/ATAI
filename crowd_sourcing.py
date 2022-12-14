import pandas as pd
import os
import rdflib


def crowd_sourcing(cro_data, URL_enti, ent2lbl):
    
    wd = rdflib.Namespace('http://www.wikidata.org/entity/')

    _, tempfilename = os.path.split(URL_enti)
    entity = 'wd:' + tempfilename
    
    answer = "None"
    support = None
    reject = None
    interatio = None

    for ind in cro_data.index:

        if entity == cro_data['Input1ID'][ind]:
            answer = str(cro_data['Input3ID'][ind])
            support = str(cro_data['support'][ind])
            reject = str(cro_data['reject'][ind])
            interatio = str(cro_data['inter-rate'][ind])
            break

    if answer[0] == 'w':
        answer = ent2lbl[wd[answer.replace('wd:','')]]
    elif answer[0] == 'Q':
        answer = ent2lbl[wd[answer]]

    return answer,support,reject,interatio

def js_search(json_object, name):
        return [obj for obj in json_object if name in obj['cast']][0]['img']

