from typing import List
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import (
	RAGenerator,
	DensePassageRetriever,
	PreProcessor
)

from haystack.utils import convert_files_to_docs, print_answers

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
    split_overlap=0
	)

	docs = convert_files_to_docs(dir_path=dir_path)
	for doc in docs:
		final_docs.append(processor.process(doc))
	
	return final_docs

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    embed_title=True,
)

document_store.update_embeddings(retriever=retriever)

generator = RAGenerator(
    model_name_or_path="facebook/rag-token-nq",
    top_k=1,
    max_length=200,
    min_length=2,
    embed_title=True,
    num_beams=2,
)


if __name__ == "__main__":
	documents = create_documents(dir_path="./pdfs")

	document_store.write_documents(documents)

	# Add documents embeddings to index
	

	pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

	questions = [
	"What is the fur trade?",
	"What are the numbered treaties?",
	"What is the manitoba act?"
	]

	for question in questions:
		print("Question: {}".format(question))
		res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
		print_answers(res, details="minimum")