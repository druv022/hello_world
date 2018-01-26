import time

start_time = time.time()
# with open('./HW1/ap_88_89/topics_title', 'r') as f_topics:
#     queries = parse_topics([f_topics])
#
# index = pyndri.Index('./HW1/index/')

num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}

query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

print('Gathering statistics about', len(query_term_ids), 'terms.')

# inverted index creation.

document_lengths = {}
unique_terms_per_document = {}

inverted_index = collections.defaultdict(dict)
collection_frequencies = collections.defaultdict(int)

total_terms = 0
ext_doc_ids = {}
ext_to_int_doc_ids = {}

for int_doc_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, doc_token_ids = index.document(int_doc_id)

    # external doc_ids -> Dictionary (key=int_doc_id, value=ext_doc_id)
    ext_doc_ids[int_doc_id] = ext_doc_id
    # Mapping external doc_id to internal doc_id
    ext_to_int_doc_ids[ext_doc_id] = int_doc_id

    document_bow = collections.Counter(
        token_id for token_id in doc_token_ids
        if token_id > 0)
    document_length = sum(document_bow.values())

    # Size of each document key->doc_id , value->length
    document_lengths[int_doc_id] = document_length

    # Total_terms : Sum over document_lengths
    total_terms += document_length

    unique_terms_per_document[int_doc_id] = len(document_bow)
    for query_term_id in query_term_ids:
        assert query_term_id is not None

        # document_term_frequency = no of times query_term_id occured in document
        document_term_frequency = document_bow.get(query_term_id, 0)

        if document_term_frequency == 0:
            continue

        collection_frequencies[query_term_id] += document_term_frequency

        inverted_index[query_term_id][int_doc_id] = document_term_frequency

avg_doc_length = total_terms / num_documents
total_docs = index.maximum_document() - index.document_base()

print('Inverted index creation took', time.time() - start_time, 'seconds.')


def run_retrieval(model_name, score_fn):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example)
    """
    run_out_path = '/home/nishant/Templates/SSaUD/output_files/{}.run'.format(model_name)

    retrieval_start_time = time.time()

    print('Retrieving using', model_name)

    data = {}

    # The dictionary data has the form: query_id --> (document_score, external_doc_id)
    for int_doc in range():
        for query_id,query_id_tokens in tokenized_queries.item():
            doc_score = [score_fn(int_doc,query_term_id,param) for query_term_id in query_id_tokens]

            query_doc_rank[query_id].append(sum(doc_score),index_document(int_doc)[0])

    print('Retrieval took', time.time() - retrieval_start_time, 'seconds.')

    with open(run_out_path, 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=1000)

------------------------------------------------------------------------------------------------------------------------
# Pre-processing
# evaluate document clusters
from functools import reduce
from sklearn.cluster import KMeans

def process_doc(model):

    # calculate average query length
    avg_query_length = reduce(lambda x,y: x+y,[len(i) for i in tokenized_queries.values()])/len(tokenized_queries.keys())

    document_cluster = {}
    document_w2v = {}
    # iterate over every document in the collection and identify the different cluster
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)

        # evaluate the number of cluster within a document on average depending on number of average query word
        # increment this if document length is more than average document length
        cluster_size = avg_query_length + int(document_lengths[int_doc_id]/avg_doc_length)

        doc_id_w2v = []

        # TODO: Remove magic number
        doc_id_w2v_avg = np.zeros(300)

        # transform words on the document to its vector representation
        for doc_token_id in doc_token_ids:
            doc_id_w2v.append(model.wv[doc_token_id])
            doc_id_w2v_avg += model.wv[doc_token_id]


        doc_kmeans = KMeans(n_clusters=avg_query_length, random_state=0).fit(np.asarray(doc_id_w2v))

        # TODO: decide ext_doc_id or int_doc_id
        document_cluster[int_doc_id] = doc_kmeans.cluster_centers_
        document_w2v[int_doc_id] = doc_id_w2v_avg

        return document_cluster,document_w2v


def word_embedding_score(int_doc_id,query_term_id,model, cluster_v = False,score_type_cos = True):

    query_id_w2v = model.wv[query_term_id]

    score = 0
    if not cluster_v:
        doc_w2v_avg = document_w2v[int_doc_id]
        score = 1 - spatial.distance.cosine(query_id_w2v,doc_w2v_avg)
    else:
        for doc_mean in document_cluster[int_doc_id]:
            if score_type_cos:
                score += 1 - spatial.distance.cosine(query_id_w2v,doc_mean)
            else:
                score += np.dot(query_id_w2v,doc_mean)

    return score








