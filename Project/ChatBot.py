import os
import re
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

vector_path = "./vectors/chroma_db"
model_name = "all-mpnet-base-v2"
model_path = "../../../Model/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
encode_kwargs={'batch_size': 64, 'normalize_embeddings': True}
LLM_MODEL_NAME = "../../../Model/TinyLlama-1_1B-Chat-V1_0"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)
    return vectorstore

QUERY_EXAMLPES = [
    {
        "q": "Кто такой Бэрдоне Оглы?",
        "a": "Бардонэ Оглы - лидер Крипонской ячейки сопротивления. Занял свой пост после смерти старшего брата."
    },
    {
        "q": "Что такое Кашкай?",
        "a": "Кашкай - Крипонская модификация Карпаннской серийной модели Дырзо. В отличии от Карпаннским моделей, Кашкай всегда оснащается корпусным пулеметом."
    }
]

MAIN_PROMPT = (
    "Ты - детектор информации. Твоя задача - найти точный ответ в контексте.  "
    "ЖЕСТКИЕ ПРАВИЛА:"
    "1. Отвечай ТОЛЬКО если информация ЕСТЬ в контексте"
    "2. Если информации НЕТ - говори: 'Я не знаю, у меня лапки'"
    "3. НЕ ПРИДУМЫВАЙ информацию"
)

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.01,
        top_p=0.1,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)

def generate_prompt(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    exmaples_text = "\n".join([f"Q: {ex['q']}\nA: {ex['a']}" for ex in QUERY_EXAMLPES])

    prompt_template = f"""
{MAIN_PROMPT}

Примеры:
{exmaples_text}

Контекст:
{{context}}

Вопрос: {{question}}

Ответ:"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    base_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return base_chain

def rag_chat():
    print("Введите вопрос")

    vectorstore = load_vectorstore()
    llm = load_llm()
    rag_chain = generate_prompt(vectorstore, llm)

    while True:
        try:
            query = input("> ").strip()
            if not query:
               continue
            response = rag_chain.invoke(query)
            print(f"Ответ: {response}")
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    rag_chat()