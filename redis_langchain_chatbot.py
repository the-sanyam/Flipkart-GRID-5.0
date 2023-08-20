
# !pip install langchain==0.0.123
# !pip install redis==4.5.3
# !pip install openai==0.27.2
# !pip install numpy
# !pip install pandas

import pandas as pd


# Load Product data 
all_prods_df = pd.read_json("/home/sanyam/Downloads/redis chatbot/flipkart_fashion_products_dataset.json")
all_prods_df.dropna(subset=["images","url"], inplace=True)

all_prods_df=all_prods_df.iloc[::30]


all_prods_df['description'] = all_prods_df.apply(lambda row: f'description:{row["description"]},title:{row["title"]},cost:{row["selling_price"]},category:{row["category"]},brand:{row["brand"]}', axis=1)
all_prods_df['images'] = all_prods_df['images'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
all_prods_df['url']=all_prods_df.apply(lambda row: [row["url"],row["images"]], axis=1)
all_prods_df=all_prods_df[["description","url"]]
# all_prods_df.drop(columns=["title",""], inplace=True)
# print(all_prods_df)

# Reset pandas dataframe index
all_prods_df.reset_index(drop=True, inplace=True)

# Num products to use (subset)
NUMBER_PRODUCTS = 10

product_metadata = (
    all_prods_df
     .head(NUMBER_PRODUCTS)
     .to_dict(orient='index')
)

# Check one of the products
# print(product_metadata[0])

# IMPORTANT : Alternative to set up redis connection, if you're not able to do it locally (decrease number of items in case of memory issue)

# import redis
# redis_conn = redis.Redis(
#   host='redis-17223.c294.ap-northeast-1-2.ec2.cloud.redislabs.com',
#   port=17223,
#   password='QaoTdtXCEMrUa15TmOtWwHGozwe7rB4E')
# client = redis.Redis(host = 'localhost', port=6379)
  
# redis_conn.ping()

import os
 
from langchain.embeddings import OpenAIEmbeddings
 
# set your openAI api key as an environment variable
os.environ['OPENAI_API_KEY'] = "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"
#(Spare key: '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')

# data that will be embedded and converted to vectors
texts = [
    v['description'] for k, v in product_metadata.items()
]

# product metadata that we'll store along our vectors
metadatas = list(product_metadata.values())
 
# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()
 
# name of the Redis search index to create
index_name = "products"
 
# assumes you have a redis stack server running on local host
redis_url="redis://localhost:6379"

# Alternate link to connect to redis server
# redis_url = "redis://default:QaoTdtXCEMrUa15TmOtWwHGozwe7rB4E@redis-17223.c294.ap-northeast-1-2.ec2.cloud.redislabs.com:17223"
from langchain.vectorstores.redis import Redis 

#All the embeddings are stored in a vectorstore

vectorstore = Redis.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embedding,
    index_name=index_name,
    redis_url=redis_url
)

print("Embeddings formed")

import os
from langchain import PromptTemplate, OpenAI, LLMChain
# from redis_langchain_chatbot import chatbot,chat_history
import chainlit as cl

from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

   
import json
from langchain.schema import BaseRetriever
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from pydantic import BaseModel
from langchain.memory import ConversationTokenBufferMemory
 

url_list=[]
class RedisProductRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore
 
    class Config:
        arbitrary_types_allowed = True
 
    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        try:
            url_list.append(metadata["url"])
        except:
            pass
        return (
        #    "Item Name: " + metadata["title"] + ". " +
           "Item Description: " + metadata["description"] + ". " 
        #    "Item Keywords: " + metadata["product_details"] + "."
        )
    
    def get_relevant_documents(self, query):
        k=0
        docs = []
        for doc in self.vectorstore.similarity_search(query):
            k+=1
            if(k==3):
                break
            content = self.combine_metadata(doc)
            docs.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))
 
        return docs

template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done. if the follow up question is not related to fashion, just say you don't know or it's not available and end the conversation. Don't make up an answer on your own.
if the follow up question is totally different from previous coversation, check following:
first, if it is not related to fashion: say you don't know or it's not available and end the conversation.
second, if it is related to fashion: continue.
if someone asks you something like "what is the weather like today?" or "who is president of usa", you can say "I don't know, but I can help you with fashion related questions."
But if someone asks you something like "what outfit will suit me" or "what should i wear on new year" or "what is trending outfit" . Answer them.

Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""
 
condense_question_prompt = PromptTemplate.from_template(template)
 
template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions,cost,url and keywords to show the shopper whats available, and answer only if it is related to fashion otherwise just ask them you can't find that currently.
 If you dont't know the answer or feel that it's not related to fashion, just say you don't know and this is not available. Don't make up an answer on your own.

 Prices/Cost are in indian rupees(Rs), don't mention any url.

 Give your answers in this format, in multiple paragraphs, in short and sweet manner(less than 100 words):
 
 Product Name:
 Cost:

 give a reason why you recommended this product?
 write a funny line to make the shopper laugh.

 Context:
{context}

Question:
 
Helpful Answer:

"""
 


@cl.on_chat_start
def init(): 
    # define two LLM models from OpenAI
    llm = OpenAI(temperature=0)

    qa_prompt= PromptTemplate.from_template(template)
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=BaseCallbackManager([
            StreamingStdOutCallbackHandler()]),
        verbose=True,
        temperature=0.4,
        max_tokens=200
    )
    memory = ConversationTokenBufferMemory(llm=streaming_llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=2000)


    # use the LLM Chain to create a question creation chain
    question_generator = LLMChain(
        llm=llm,
        prompt=condense_question_prompt
    )

    # use the streaming LLM to create a question answering chain
    doc_chain = load_qa_chain(
        llm=streaming_llm,
        chain_type="stuff",
        prompt=qa_prompt
    )

        
    redis_product_retriever = RedisProductRetriever(vectorstore=vectorstore)
    
    chain = ConversationalRetrievalChain(
        
        retriever=redis_product_retriever,
        combine_docs_chain=doc_chain,
        memory=memory,
        verbose=True,
        rephrase_question=False,
        question_generator=question_generator
    )
    
    cl.user_session.set("conversation_chain", chain)
 

@cl.on_message
async def main(message: str):
        
        
        url_list.clear()

 
        chain = cl.user_session.get("conversation_chain")
        
        res = chain.run({"question":message},callbacks=[cl.LangchainCallbackHandler()])
        

        elements=[]
        s=set()
        for item in url_list:
            if(item[0] not in s):
                s.add(item[0])
                elements.append(cl.Image(name="image1",display="inline",url=item[1]))
        
        # Send the answer and the text elements to the UI
        await cl.Message(content=res,elements=elements).send()