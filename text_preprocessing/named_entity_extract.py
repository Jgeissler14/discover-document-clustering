import spacy
import re
from nltk import defaultdict
import numpy as np

def get_named_entity_counts(nlp, raw_text):
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

# Function to create a list of similarity ratings for the most common entities between docs
def get_entity_similarities(nlp, top_k_words, pdf_entities_1, pdf_entities_2): 
    
    # Initialize empty lists to house entity similarity comparisons
    entity_similarities = list() 
    num_duplicate_entities = int()
    no_vector_ents = list()

    # List comprehension to get the "n" most common entities from each doc
    doc_1_entities = [term_1 for term_1, _ , _ , _ in pdf_entities_1[:top_k_words]]
    doc_2_entities = [term_2 for term_2, _ , _ , _ in pdf_entities_2[:top_k_words]]

    # Create lists of doc objects in bulk with nlp.pipe
    ents_1 = list(nlp.pipe(doc_1_entities))
    ents_2 = list(nlp.pipe(doc_2_entities))
    print(ents_1, ents_2)
    
    idx_list = list()
    idy_list = list()
    duplicated_ents = list()
    
    
    # Remove entities without word vectors:
    for idx, x in enumerate(ents_1):
    # Check if entity has a vector; if not, remove it from the list
        if x.vector_norm == 0:
            print('*** no vector found', x)
            ents_1.pop(idx)
            no_vector_ents.append(x)
    for idy, y in enumerate(ents_2):
        # Check if entity has a vector; if not, remove it from the list
        if y.vector_norm == 0:
            print('*** no vector found', y)
            ents_2.pop(idy)
            no_vector_ents.append(y)
    
    # Find duplicate entities from both docs
    for idx, x in enumerate(ents_1):
        for idy, y in enumerate(ents_2):
            # Check if entities are duplicates       
            if x.text == y.text and x.similarity(y) == 1.0:
                # If so, add the indexes to two lists
                idx_list.append(idx)
                idy_list.append(idy)
                duplicated_ents.append(x)

    # Reverse the sorting
    idx_list.sort(reverse=True)
    idy_list.sort(reverse=True)
    
    num_duplicate_entities = len(duplicated_ents)
    
    print('no_vector_ents ... ', no_vector_ents)
    print('duplicated_ents ... ', duplicated_ents, len(duplicated_ents))
    
    # Loop through both lists to get similarity ratings between the two lists of entities
    for f in ents_1:
        for b in ents_2:
            # print(f, b, f.similarity(b))
            entity_similarities.append(f.similarity(b))
    
    return entity_similarities, num_duplicate_entities, no_vector_ents

# Function to get the log frequency products between the two lists 
def get_entity_log_freqs(nlp, top_k_words, pdf_entities_1, pdf_entities_2):    
    
    # Initialize empty list to house entity log_frequency (normalized) values
    entity_log_freqs = list()
    
    doc_1_log_freqs = [log_freq_1 for _, _ , _ , log_freq_1 in pdf_entities_1[:top_k_words]]
    doc_2_log_freqs = [log_freq_2 for _, _ , _ , log_freq_2 in pdf_entities_2[:top_k_words]]
    
    for x in doc_1_log_freqs:
        for y in doc_2_log_freqs:
            entity_log_freqs.append(x * y)
    
    return entity_log_freqs