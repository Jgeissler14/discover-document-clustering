import spacy
import re
from nltk import defaultdict
import numpy as np

nlp = spacy.load('en_core_web_lg')

def get_named_entity_counts(raw_text):
    # Create spacy doc
    doc = nlp(raw_text)
    
    # Store the number of words in the doc
    tot_num_words = len(doc)

    # Create list of named entities along with their labels
    doc_entities = [[re.sub('  ', ' ', re.sub('\r',' ',re.sub('\n', ' ', ent.text))).strip().lower(), ent.label_, ] 
                    for ent in doc.ents if ent.text.isnumeric() == False and ent.text.startswith('https://') == False] 
    
    # Create defaultdict to list the entities and their counts
    entities_cnt = defaultdict(int)
    
    # Create frequency count for each of the entities
    for entity, ent_label in doc_entities:
        entities_cnt[entity] +=1

    # Sort the entities list by the number of occurrences 
    entities_cnt_sorted = sorted(entities_cnt.items(), key=lambda x: x[1], reverse=True)
    # print(entities_cnt_sorted)
    
    # Get the entity frequency list (including counts, entity cnt / total doc length, & log frequency 
    ent_doc_freq_list = [[entity[0],  entity[1], entity[1]/tot_num_words, np.log(1 + int(entity[1]))] 
                         for entity in entities_cnt_sorted]
 
    # print(ent_doc_freq_list)
        
    # Also get a simple list of just the entity names
    extracted_entities = [entity[0] for entity in entities_cnt_sorted]
    entity_cnt =    [entity[1] for entity in entities_cnt_sorted]
    entity_freq = [entity[1]/tot_num_words for entity in entities_cnt_sorted]
    entity_log_freq = [np.log(1 + int(entity[1])) for entity in entities_cnt_sorted]
    
    # print(extracted_entities) 

    return ent_doc_freq_list