from transformers import pipeline, set_seed
import joblib
from transformers import AutoTokenizer, AutoModelForTokenClassification

"""
The NER based on the Bert transformer.
"""
def save_transformer():

    set_seed(111)
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    joblib.dump(nlp, 'NER_trans.pkl')

# run the trans_pred to get the entities of a question.

def trans_pred(question, filepath = 'NER_trans.pkl'):

    que = []

    # double check the question, omit those words very unlikely to be the entity"
    omit_list = ['Who','What','what','is','was','are','were','does','you','me','name','movie','film',
    'show','Show','picture','Picture','look','looks']

    token_list = question.split(' ')
    print(token_list)
    for token in token_list:
        if not token in omit_list:
            if(not 'film' in token) and (not 'movie' in token):
                que.append(token)

    que = ' '.join(que)
    print(que)

    entities = []
    ner_pipeline = joblib.load(filepath)
    ner = ner_pipeline(que, aggregation_strategy = "simple")
    for entity in ner: 
        entities.append(entity['word'])

    return entities

