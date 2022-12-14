import spacy
from spacy.lang.en import English


def getSentences(text):
    nlp = English()
    config = {"punct_chars": None}
    nlp.add_pipe("sentencizer", config=config)
    document = nlp(text)
    return [sent.text.strip() for sent in document.sents]
    print('Get tokens Done')

def isRelationCandidate(token):
    # deps = ["ROOT", "adj", "attr", "agent", "amod"]
    deps = ["ROOT", "attr","adj","agent", "amod","ADJ",'aux','nsubj','ccomp','compound','dobj','pobj']
    return any(subs in token.dep_ for subs in deps)

def appendChunk(original, chunk):
    if original == '': return chunk
    return original + ' ' + chunk

def get_relations(question):

    relation = ''

    nlp_model = spacy.load('en_core_web_sm')
    sentence = getSentences(question)
    tokens = nlp_model(sentence[0])
    # print(tokens)

    for token in tokens:
        print(token.text, token.dep_)
        if len(token)>=2 :
            if not (token.text[0].isupper() and token.text[1].islower()): 
                if isRelationCandidate(token):
                    omit_list = ['Who','What','what','is','was','are','were','does','you','me','movie','name']
                    if not token.text in omit_list:
                        relation = appendChunk(relation, token.text)
    return relation

