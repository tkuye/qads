from typing import List
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import ExtractiveQAPipeline

from haystack.nodes import (
	FARMReader,
	DensePassageRetriever,
	PreProcessor
)

from haystack.utils import convert_files_to_dicts, print_answers, launch_milvus
from haystack.document_stores import MilvusDocumentStore


def create_documents(dir_path) -> List[Document]:
	"""
		Creates a list of documents from a directory of pdfs.
	"""

	final_docs = []
	processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
    split_overlap=10
	)

	docs = convert_files_to_dicts(dir_path=dir_path)
	for doc in docs:
		final_docs.extend(processor.process(doc))
	
	return final_docs




documents = create_documents(dir_path="./pdfs")

launch_milvus()
document_store = MilvusDocumentStore()

document_store.write_documents(documents)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    embed_title=True,
    use_fast_tokenizers=True,
)

#document_store.update_embeddings(retriever=retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")


if __name__ == "__main__":
	
	pipe = ExtractiveQAPipeline(reader, retriever)

	questions = [
	"What is the fur trade?",
	"What are the numbered treaties?",
	"What is the manitoba act?"
	]

	for question in questions:
		res = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
		print_answers(res, details="minimum")