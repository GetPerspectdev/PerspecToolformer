from langchain import PromptTemplate, LLMChain, LlamaCpp
import chainlit as cl

from datasets import load_dataset

import json

from gpt4all import Embed4All
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
import asyncio

import os

import nest_asyncio
nest_asyncio.apply()

number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ###Instruction:
    You are an expert witness specializing in empathy, toxicity, and professionalism.
    Given a person's message history and a current message as context, rate the messages on a scale of 1-100 for how professional they are (higher scores indicate more professional messages).
    Please respond with only an integer between 1 and 100, then give a short explanation of how the person could be more professional.

    ###Input:
    Message History: 

    Current Message:
    {context}


    ###Response:
    Your Professionalism rating from 1-100 is """

llm = LlamaCpp(
            model_path="./models/losslessmegacoder-llama2-13b-q2_k.bin",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=8192,
            verbose=True
            )

# Utils
class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        # chunks = []
        # for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
        #     chunks.append(text[i : i + self.chunk_size])
        return text.split("\n")

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class DataLoader:
    def __init__(self, embedding_model: Embed4All = None, data_path: str = None):
        self.ds = load_dataset('csv', data_files=data_path, split="train")
        self.embedder = embedding_model or Embed4All()

    def embed_texts(self, examples):
        embedding = self.embedder.embed(examples['text'])
        return {"embedding": embedding}

    def embed_ds_and_index(self):
        emb_ds = self.ds.map(self.embed_texts, batched=False)
        emb_ds.to_csv("./data/embedded_dataset.csv")
        emb_ds.add_faiss_index("embedding")
        emb_ds.save_faiss_index("embedding", "professionalism_index.faiss")

    def load_index(self, index):
        emb_ds = self.load_dataset('csv', data_files="./data/embedded_dataset.csv", split='train')
        emb_ds.load_faiss_index('embedding', './data/professionalism_index.faiss')
        return emb_ds

class VectorDatabase:
    def __init__(self, embedding_model: Embed4All = None, DB=None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or Embed4All()
        self.verbose = True
        self.DB = DB

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search_by_text(
        self,
        query_text: str,
        k: int,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        if self.verbose:
            print(f"Embedding {query_text[:10]}...")
        query_embed = self.embedding_model.embed(query_text)
        query_vector = np.array(query_embed, dtype=np.float32)
        score, samples = self.DB.get_nearest_examples('embedding', query_vector, k=k)
        ratings = [f"{sample['rating']}, {sample['Comment']}" for sample in samples]
        return [rating[0] for rating in ratings] if return_as_text else ratings

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: LlamaCpp, vector_db_retriever: VectorDatabase, vector_db_loader: DataLoader, template=None, verbose=False) -> None:
        self.llm = llm
        self.template = template
        self.vector_db_retriever = vector_db_retriever
        self.vector_db_loader = vector_db_loader
        self.verbose = verbose

    def run_pipeline(self, user_query: str) -> str:
        if self.verbose:
            print("Loading ")

        if self.verbose:
            print(f"Searching VectorDB for {user_query[:10]}...")
        context_list = self.vector_db_retriever.search_by_text(user_query, k=2)
        
        if self.verbose:
            print("Gathering context...")
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_prompt_template = PromptTemplate(input_variables=['message_history', 'context'], template=self.template)
        chain = LLMChain(llm=self.llm, prompt=formatted_prompt_template)
        if self.verbose:
            print("Running Chain")

        professionalism = chain.run({"message_history":user_query, "context": context_prompt})
        obj = json.dumps({"professionalism": professionalism}, indent=4)
        return obj

@cl.on_chat_start
def main():
    #Instantiate the chain
    prompt = PromptTemplate(template=number_template, input_variables=["context"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain") #type: LLMChain

    # Call the chain asyncronously
    res = await cl.make_async(llm_chain)(message, callbacks=[cl.LangchainCallbackHandler()])

    # Do post processing and RAG

    # 'res' is a Dict
    await cl.Message(content=res['text']).send()
    return llm_chain