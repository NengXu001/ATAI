from rdflib.namespace import RDFS
from rdflib.term import URIRef
import rdflib
import re
import editdistance

def preprocess_graph(filepath):

    graph = rdflib.Graph()
    graph.parse(filepath, format='turtle')

    g = graph
    nodes = {}
    predicates = {}

    for node in g.all_nodes():
        if isinstance(node, URIRef):
            if g.value(node, RDFS.label):
                nodes[node.toPython()] = g.value(node, RDFS.label).toPython()
            else:
                nodes[node.toPython()] = node.toPython()

    for s, p, o in g:
        if g.value(p, RDFS.label):
            predicates[p.toPython()] = g.value(p, RDFS.label).toPython()
        else:
            predicates[p.toPython()] = re.sub('http://www.wikidata.org/entity/', "", p.toPython())
    
    return g,nodes,predicates


def test_dis(nodes,predicates,entity,relation):

    tmp = 12
    match_node = []
    for key, value in nodes.items():
        if editdistance.eval(value, entity) <= tmp:
            tmp = editdistance.eval(value, entity)
            match_node.append([key,value]) 
            if len(match_node) >= 3:
                match_node = match_node[-2:]
    match_node.reverse()

    match_pred = []
    tmp = 15
    for key, value in predicates.items():
        if editdistance.eval(value, relation) <= tmp:
            tmp = editdistance.eval(value, relation)
            match_pred.append([key,value])
            if len(match_pred) >= 1:
                match_pred = match_pred[-1:]

    print("\n--- the matching node of \"{}\" is {}\n".format(entity, match_node))
    print("--- the matching predicates of \"{}\" is {}\n".format(relation, match_pred))

    return match_node, match_pred

def SPA(g, match_node, match_pred):

    answer = None

    for node in match_node:

        query_template = "SELECT ?label WHERE {{ <{}> <{}> / <{}> ?label. FILTER(LANG(?label) = 'en').}}".format(node[0], match_pred[0][0],  RDFS.label)
        qres = g.query(query_template)

        for row in qres:
            
            print(row.label.toPython())

            final_node = node[1]
            answer = row.label.toPython()
            
            return answer,final_node,match_pred[0][1]

    return answer,None,None

def SPA_image(g, match_node, match_pred):

    answer = None

    node = match_node[0]

    query_template = "SELECT ?label WHERE {{ <{}> <{}> ?label}}".format(node[0], match_pred[0][0])
    qres = g.query(query_template)

    for row in qres:

        final_node = node[1]
        answer = row.label.toPython()
        
        return answer,final_node,match_pred[0][1]

    return answer,None,None