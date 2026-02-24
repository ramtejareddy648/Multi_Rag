from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_groq import ChatGroq

from typing import List,Dict,Annotated
import hashlib
from langchain_core.messages import HumanMessage
import base64
from langchain_openai import ChatOpenAI

import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pickle
from typing import List,TypedDict
from pydantic import BaseModel,Field
import re
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import START,StateGraph,END
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
# import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_astradb import AstraDBVectorStore, AstraDBByteStore
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
import streamlit as st
# load_dotenv()

if "STREAMLIT_RUNTIME" not in os.environ:
    load_dotenv()



DB_URI = os.getenv("POSTGRES_URL")


@st.cache_resource
def get_connection_pool():
    
    db_url = os.environ.get("POSTGRES_URL") 
    return ConnectionPool(
        conninfo=db_url,
        max_size=10,    
        min_size=1,     
        timeout=30.0,   
        kwargs={
            "autocommit": True,
            "sslmode": "require"
        },
    )

pool = get_connection_pool()
check_point = PostgresSaver(pool)



def initialize_db():
    try:
        
        check_point.setup() 
        print("--- NEON CLOUD TABLES INITIALIZED ---")
    except Exception as e:
        
        print(f"Database Init Info: {e}")


initialize_db()

try:
    import fitz
    PYMUPDF_AVALIBLE=True
except ImportError:
    PYMUPDF_AVALIBLE=False
    print('warning pymupdf is import')
    
    

class State(TypedDict):
    question:str
    answer:str
    messages:Annotated[List[BaseMessage], add_messages]
    docs:List[Document]
    good_docs:List[Document]
    verdict:str
    reason:str
    web_query:str
    web_docs:List[Document]
    strips:List[str]
    keep_strips:List[str]
    refined_context_docs:List[Document]
    next:str



class Docvalided(BaseModel):
    score: float = Field(description="Relevance score between 0.0 and 1.0")
    reason: str = Field(description="Brief explanation for the score")


class KeeporDrop(BaseModel):
    keep:bool


class WebQuery(BaseModel):
    query:str

class CustomMultiVectorRetriever:
    def __init__(self, vectorstore, parent_store, k=6):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.k = k

    def get_relevant_documents(self, query):
        chunks = self.vectorstore.similarity_search(query, k=self.k)
        doc_ids = [chunk.metadata.get("doc_id") for chunk in chunks if chunk.metadata.get("doc_id")]

        if not doc_ids:
            return []

    
        retrieved_bytes = self.parent_store.mget(doc_ids)
    
    
        parents = []
        for b in retrieved_bytes:
            if b is not None:
                parents.append(pickle.loads(b)) 
            
        return parents
    
    
    
class MultiRag:
    def __init__(self):
        self.endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.embed_model=OpenAIEmbeddings(
            model="text-embedding-3-small",
            timeout=60
        )
        
        
        self.vectors = AstraDBVectorStore(
        embedding=self.embed_model,
        collection_name=os.getenv("VECTOR_COLLECTION_NAME", "rag_vector"),
        api_endpoint=self.endpoint,
        token=self.token,
        )
        
        
        self.parent_store = AstraDBByteStore(
        api_endpoint=self.endpoint,
        token=self.token,
        collection_name=os.getenv("PARENT_STORE_COLLECTION_NAME", "rag_parents"),
        )
        
        self.text_spliter=RecursiveCharacterTextSplitter(
            chunk_size=1500,  
            chunk_overlap=300, 
        )
        
        self.llm=ChatGroq(model='llama-3.3-70b-versatile')
        
        
        self.vision_model=ChatOpenAI(model='gpt-4o-mini')
        # self.travily=TavilySearchResults(max_results=5)
        self.web_tool = DuckDuckGoSearchAPIWrapper(max_results=5)
        
        self.retriever = CustomMultiVectorRetriever(
        vectorstore=self.vectors,
        parent_store=self.parent_store,
        k=6
        )
        
    
    
    
    
    
    
    
        
        
        
    def step_multi_vector_store(self):
        
        """
        Ensures the AstraDB connections are active before ingestion or retrieval.
        """
        if not self.vectors:
            self.vectors = AstraDBVectorStore(
            embedding=self.embed_model,
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", "rag_vector"),
            api_endpoint=self.endpoint,
            token=self.token,
        )
    
        if not self.retriever:
            self.retriever = CustomMultiVectorRetriever(
            vectorstore=self.vectors,
            parent_store=self.parent_store,
            k=6
        )
    def parse_documents(self,file_paths:List[str])->Dict[str,List]:
        
        """
            load documents like pdf text store like images texts and tables
        """
        
        all_elements={'texts':[],'tables':[],'images':[]}
        
        for file_path in file_paths:
            try:
                if file_path.endswith('pdf'):
                    
                    loader=PyPDFLoader(file_path)
                    docs=loader.load()
                    for doc in docs:
                        all_elements['texts'].append({
                        'content':doc.page_content,
                        'metadata':{'source':file_path,'type':'text','page':doc.metadata.get('page',0)}
                    
                    
                        })
            
                        if PYMUPDF_AVALIBLE:
                            pdf_images=self._extract_images_from_pdf(file_path)
                            all_elements['images'].extend(pdf_images)
                        
                        else:
                            print('PYMUPDF is not avalible in side parse function')
                
                elif file_path.endswith('txt'):
                     loader=TextLoader(file_path)
                     docs=loader.load()
                     
                     for doc in docs:
                         all_elements['texts'].append({
                             'content':doc.page_content,
                             'metadata':{'source':file_path,'type':'text','page':0}
                             
                         })
                else:
                    continue
                    
                            
            except Exception as e:
                print('error inside except block of parse documents')
        
        
        return all_elements
    
    
    
    def _extract_images_from_pdf(self,file_path):
        """
        Extract images from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted image data
        """
        
        extracted_images=[]
        
        if not PYMUPDF_AVALIBLE:
            print("PyMuPDF not available - cannot extract images from PDF")
            return extracted_images
        
        
        try:
            
            doc=fitz.open(file_path)
            
            
            for page_num in range(len(doc)):
                page=doc[page_num]
                image_list=page.get_images()
                
                
                for img_indx,img in enumerate(image_list):
                    
                    xref=img[0]
                    base_image=doc.extract_image(xref)
                    image_bytes=base_image['image']
                    image_ext=base_image['ext']
                    
                    
                    image_hash=hashlib.md5(image_bytes).hexdigest()
                    
                    
                    extracted_images.append({
                        'content':image_bytes,
                        'metadata':{
                            'source':file_path,
                            'type':'image',
                            'page':page_num+1,
                            'image_index':img_indx,
                            'extension':image_ext,
                            'hash':image_hash
                        }
                        
                    })
                    
                    print(f"Extracted image {img_indx + 1} from page {page_num + 1} of {file_path}")
            
            doc.close()
        
        
        except Exception as e:
            print(f"Error extracting images from PDF {file_path}: {str(e)}")
        
        
        
        return extracted_images
    
    
    def process_images_advanced(self,image_paths:List[str])->Dict[str,List]:
        """_summary_

        Args:
            image_paths (List[str]): _description_

        Returns:
            Dict[str,List]: _description_
        """
        images_discrption=[]
        for image_path in image_paths:
            try:
                with open(image_path,'rb') as file:
                    image_data=file.read()
                    
                    hash_image=hashlib.md5(image_data).hexdigest()
                    
                    description = self._generate_enhanced_image_description(image_data)
                    
                    text_content = self._extract_text_from_image(image_data)
                    
                    images_discrption.append({
                        'path':image_path,
                        'hash':hash_image,
                        'description':description,
                        'contex':text_content,
                        'type':"image",
                        'metadata':{
                            'source':image_path,
                            'type':'image',
                            'hash':hash_image
                        }
                    })
            
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
        return images_discrption
    
    def _generate_enhanced_image_description(self,image_data:bytes)->str:
        """generate image description based on image bytes to openai vision model"""
        image_b64=base64.b64encode(image_data).decode()
        prompt = """
            Analyze this image comprehensively and provide a detailed description that includes:
            
            1. **Main Content**: Objects, people, scenes, activities
            2. **Visual Elements**: Colors, composition, style, lighting
            3. **Text Content**: Any visible text, signs, labels, captions
            4. **Context & Setting**: Environment, location, time period if apparent
            5. **Technical Details**: Charts, graphs, diagrams, technical elements
            6. **Relationships**: How elements in the image relate to each other
            7. **Actionable Information**: Key insights or data points visible
            
            Format your response to be detailed yet concise, optimized for retrieval and understanding.
            Focus on information that would be valuable for question-answering scenarios.
            """
        message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
        
        
        response=self.vision_model.invoke([message])
        
        
        return response.content
    
    def _extract_text_from_image(self,imag_data:bytes)->str:
        """_summary_

        Args:
            imag_data (bytes): _description_

        Returns:
            str: _description_
        """
        
        image_64=base64.b64encode(imag_data).decode()
        
        
        prompt = """
            Extract all visible text from this image. Include:
            - Any readable text, numbers, labels
            - Text in charts, graphs, diagrams
            - Signs, captions, headers
            - Technical annotations
            
            Provide only the extracted text content, maintaining structure where possible.
            If no text is visible, respond with "No text detected".
            """
        
        message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{image_64}"
                }
            }
        ]
        )

        
        response=self.vision_model.invoke([message])
        
        
        return response.content
    
    
    
    def create_summaries_for_retrieval(self,elements:List[Dict])->List[str]:
        """
        take element and give summary of content
        """
        
        summarry=[]
        try:
            for element in elements:
                content=element.get('content')
                type=element.get('metadata',[]).get('type','image')
                
                if type=="table":
                    prompt = f"""
                    Summarize this table for optimal retrieval. Focus on:
                    - Key data points and metrics
                    - Column headers and categories
                    - Notable trends or patterns
                    - Actionable insights
                    
                    Table content:
                    {content[:2000]}  # Limit content length
                    
                    Provide a concise summary optimized for semantic search.
                    """
                
                elif type=='image':
                    summarry.append(content)
                    continue
                else:
                    prompt = f"""
                    Create a retrieval-optimized summary of this text. Include:
                    - Main topics and themes
                    - Key facts and figures
                    - Important concepts
                    - Actionable information
                    
                    Text content:
                    {content[:2000]}
                    
                    Provide a comprehensive yet concise summary.
                    """
                
                
                response=self.llm.invoke(prompt)
                
                summarry.append(response.content)
        
        
        
        except Exception as e:
            print(f'exception is {e} in function create_summaries_for_retrieval')

        
        return summarry
    
    
    
    
    def build_enhanced_vector_database(self,image_paths:List[str]=[],file_paths:List[str]=[]):
        """_summary_

        Args:
            image_paths (List[str]): _description_
            file_paths (List[str]): _description_
            
        """
        
        if not image_paths and not file_paths:
            print("No new documents provided. Using existing vector database.")
            return
    
        self.step_multi_vector_store()
        
        parsed_ele = {'texts':[], 'tables':[], 'images':[]}
        processed_images = []
        
        if file_paths:
            paesed_ele=self.parse_documents(file_paths)
        
        
        if image_paths:
        
        
            processed_images=self.process_images_advanced(image_paths)
        
        else:
            []
        
        
        pdf_extracted_image=[]
        for img_data in paesed_ele['images']:
            try:
                
                description=self._generate_enhanced_image_description(img_data['content'])
                context_text=self._extract_text_from_image(img_data['content'])
                
                processed_image={
                    'description':description,
                    'text_context':context_text,
                    'metadata':img_data['metadata']
                }
                
                pdf_extracted_image.append(processed_image)
                print(f"Processed extracted image from {img_data['metadata']['source']} (page {img_data['metadata']['page']})")
            except Exception as e:
                print(f"Error processing extracted image: {str(e)}")
        
        all_elements=[]
        
        for text_data in paesed_ele['texts']:
            all_elements.append(text_data)
        
        
        for table_data in paesed_ele['tables']:
            all_elements.append(table_data)
        
        
        for file_image in processed_images:
            all_elements.append({
                'content':file_image['description'],
                'metadata':file_image['metadata'],
                'orignal_data':file_image
            })
        
        
        for pdf_image in pdf_extracted_image:
            all_elements.append({
                'content':pdf_image['description'],
                'metadata':pdf_image['metadata'],
                'orignal_data':pdf_image
            })
        
        if not all_elements:
            print('no elements is processed in vector function')
            return 
        
        
        summaries=self.create_summaries_for_retrieval(all_elements)
        
        
        doc_ids=[str(uuid.uuid4()) for _ in all_elements]
        
        sumarry_doc=[]
        
        for i, (summary, element) in enumerate(zip(summaries,all_elements)):
            doc=Document(
                page_content=summary,
                metadata={
                    'doc_id':doc_ids[i],
                    'source':element['metadata'].get('source','uknown'),
                    'type':element['metadata'].get('type','text')
                }
                
                
            )
            sumarry_doc.append(doc)
        
        
        
        self.vectors.add_documents(sumarry_doc)
        
        orignal_doc=[]
        
        for i,element in enumerate(all_elements):
            
            ori_doc=Document(
                page_content=element['content'],
                 metadata=element['metadata']
             )
            if 'orignal_data' in element:
                ori_doc.metadata['orignal_data']=element['orignal_data']
            
           
        
            orignal_doc.append((doc_ids[i],ori_doc))

        parent_pairs = []
        for i, (_, ori_doc) in enumerate(orignal_doc):
        
            doc_bytes = pickle.dumps(ori_doc) 
            parent_pairs.append((doc_ids[i], doc_bytes))


        self.parent_store.mset(parent_pairs)
        
        
      
        
        
       
        print(f"Enhanced vector database built with {len(all_elements)} NEW elements")
        print(f"- Text elements: {len(paesed_ele['texts'])}")
        print(f"- Table elements: {len(paesed_ele['tables'])}")
        print(f"- Image elements: {len(processed_images) + len(pdf_extracted_image)}")
        print("Sync to AstraDB Cloud successful.")
        
    
    
        
    
    
    def reteriver_node(self,state:State)->State:
        """_summary_

         Args:
         query (str): _description_
             k (int, optional): _description_. Defaults to 6.

         Returns:
            List[Document]: _description_
        """
        if not self.retriever:
            print('reteriver not initalized')
            self.step_multi_vector_store()
            
        question=state['question']
        docs=self.retriever.get_relevant_documents(question)
        print(f"--- RETRIEVED {len(docs)} DOCUMENTS ---")
        return{'docs':docs} 
    
    
    
    def doc_eval_node(self,state:State)->State:
        
        
        doc_eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert relevance grader for a Retrieval-Augmented Generation system.\n"
        "Your goal is to determine if the provided context is USEFUL for answering the question.\n\n"
        "Grading Criteria:\n"
        "1. If the chunk contains keywords or concepts related to the question, it is relevant.\n"
        "2. The chunk DOES NOT need to answer the entire question alone to be considered relevant.\n"
        "3. Be liberal with relevance; if it provides a helpful lead, score it highly.\n\n"
        "Response Format:\n"
        "Return ONLY a JSON object with two keys: 'score' (float 0.0 to 1.0) and 'reason' (brief string)."
    ),
    ("human", "Question: {question}\n\nRetrieved Chunk:\n{chunk}"),
    ])
        # 
        
        llm_with_structure=self.llm.with_structured_output(Docvalided)
        
        doc_eval_chain=doc_eval_prompt | llm_with_structure
        
        LOWER_TH=0.3
        UPPER_TH=0.7
        
        question=state.get('question',' ')
        docs=state.get('docs',[])
        if not docs:
            return {
                "verdict": "INCORRECT",
                "reason": "No documents were found by the retriever.",
                "good_docs": []
            }
        
        scores = []
        good = []
    
        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            output = doc_eval_chain.invoke({'question': question, 'chunk': content})
            current_score=output.score
            scores.append(current_score)
        
            if current_score>LOWER_TH:
                
                good.append(doc)
        
        if any(s > UPPER_TH for s in scores):
            return {'good_docs': good, 'verdict': "CORRECT", 'reason': "High relevance found."}
    
        if all(s < LOWER_TH for s in scores):
            return {'good_docs': [], 'verdict': 'INCORRECT', 'reason': "All chunks irrelevant."}
    
        return {'good_docs': good, 'verdict': 'AMBIGUOUS', 'reason': "Partial relevance found."}
    
    
    def decomepose_sentences(self,text:str)->List[str]:
        text=re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip())>20]
    
    
    def refine(self,state:State)->State:
        
        
        filter_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict relevance filter. Return keep=true only if the sentence directly helps answer the question. Output JSON only."),
            ("human", "Question: {question}\n\nSentence:\n{sentence}"),
        ])
        output_parser=JsonOutputParser(pydantic_object=KeeporDrop)
        
        filter_chain=filter_prompt | self.llm | output_parser
        
        
        question=state['question']
        if state['verdict']=='CORRECT':
            use_docs=state['good_docs']
        elif state['verdict']=='INCORRECT':
            use_docs=state.get('web_docs',[])
        else:
            use_docs=state.get('web_docs',[])+state['good_docs']
    
    
        # context='\n\n'.join(doc.page_content for doc in use_docs).strip()
    
        # strips=self.decomepose_sentences(context)
    
        all_strips = []
        kept_docs = []
        kept_sentences = []

        for doc in use_docs:
            # Decompose each document into sentences
            doc_sentences = self.decomepose_sentences(doc.page_content)
            all_strips.extend(doc_sentences)
            
            for s in doc_sentences:
                output = filter_chain.invoke({'question': question, 'sentence': s})
                ans=output.get('keep','')
                if ans:
                    kept_sentences.append(s)
                    
                    new_doc = Document(
                        page_content=s,
                        metadata=doc.metadata
                    )
                    kept_docs.append(new_doc)

        return {
            'strips': all_strips,
            'keep_strips': kept_sentences,
            'refined_context_docs': kept_docs 
        }
        
    
    def rewrite_query_node(self,state:State)->State:
        
        rewrite_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
        ]
        )
        
        
        output_parser=JsonOutputParser(pydantic_object=WebQuery)


        rewrite_chain= rewrite_prompt | self.llm | output_parser
        
        question=state['question']
        output=rewrite_chain.invoke({'question':question})
        update_query=output.get('query','')
    
        return{
        'web_query':update_query
        }
    
    
    def web_search_node(self,state:State)->State:
        question=state.get('web_query') or state['question']
        results = self.web_tool.results(question, max_results=5)
    
        web_docs:List[Document]=[]
    
        for r in results:
            title = r.get("title", "No Title")
            url = r.get("link", "")      
            content = r.get("snippet", "") 
        
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        
            web_docs.append(Document(
                page_content=text, 
                metadata={"url": url, "title": title, "source": "duckduckgo"}
            ))
    
        print(f"--- WEB SEARCH COMPLETED: {len(web_docs)} RESULTS ---")
        return {'web_docs': web_docs}
    def generate(self,state: State) -> State:
        """
        Refined generate function that handles multimodal context sorting 
        and prompt formatting within the graph state.
        """
        question = state['question']
        relevant_docs = state.get('refined_context_docs', []) 
        history = state.get('messages', [])

    
        image_context = []
        table_context = []
        text_context = []
    
        for doc in relevant_docs:
            doc_type = doc.metadata.get('type', 'text')
            source = doc.metadata.get('source', 'unknown')
            
            
            if doc_type == 'image':
                image_context.append(f"Source: {source} (Visual Analysis): {doc.page_content}")
            elif doc_type == 'table':
                table_context.append(f"Source: {source} (Table Data): {doc.page_content}")
            else:
                text_context.append(f"Source: {source}: {doc.page_content}")
        
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful  assistant. Use the provided context and conversation history to answer accurately."),
            *history, 
            ("human", f"""
                Use this context to answer:
                
                TEXT CONTEXT: {" / ".join(text_context) or "None"}
                TABLE CONTEXT: {" / ".join(table_context) or "None"}
                IMAGE CONTEXT: {" / ".join(image_context) or "None"}
                
                CURRENT QUESTION: {question}
            """)
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({})
        
        return {
            "answer": response.content,
            'messages':[AIMessage(content=response.content)]
        }
    
    
    
        
        
    def route_after_eval_doc(self,state:State):
        if state['verdict']=='CORRECT':
            return 'refine'
        else:
            return 'rewrite_query'
    
    
        
def workflow_fun():
    rag=MultiRag()
    graph=StateGraph(State)
    
    graph.add_node('reteriver',rag.reteriver_node)
    graph.add_node('eval_each_doc',rag.doc_eval_node)
    graph.add_node('refine',rag.refine)
    graph.add_node('rewrite_query',rag.rewrite_query_node)
    graph.add_node('web_search',rag.web_search_node)
    graph.add_node('generate',rag.generate)
   

    graph.add_edge(START,'reteriver')
    graph.add_edge('reteriver','eval_each_doc')
    graph.add_conditional_edges(
    'eval_each_doc',
    rag.route_after_eval_doc,{
    'refine':'refine',
    'rewrite_query':'rewrite_query'
    }
    
    )
    graph.add_edge('refine','generate')
    graph.add_edge('generate',END)
    graph.add_edge('rewrite_query','web_search')
    graph.add_edge('web_search','refine')
    
        
    flow=graph.compile(checkpointer=check_point)
    
    return flow


workflow=workflow_fun()





def reterive_all_threads(user_id:str):
    all_threads = set()
    try:
       
        for checkpoint in check_point.list(None,filter={"user_id": user_id}):
            t_id = checkpoint.config['configurable'].get('thread_id')
            if t_id:
                all_threads.add(t_id)
    except Exception as e:
        print(f"Sidebar loading info: {e}")
        st.error(f"Error loading chat history: {e}")
    return list(all_threads)

 
def delete_thread(thread_id: str):
    """Delete thread data from the cloud Postgres database.""" 
    try:
        
        target_id = str(thread_id)
        
        
        tables = ['checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'checkpoint_metadata']
        
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                for table in tables:
                   
                    query = f'DELETE FROM {table} WHERE thread_id = %s'
                    cursor.execute(query, (target_id,))
                
                
                conn.commit()
        
        print(f"Successfully deleted all data for thread: {target_id}")
        return True
    except Exception as e:
        print(f"Error while deleting thread from Postgres: {e}")
        return False
 
