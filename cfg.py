from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, Completer, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import DeepLakeRetriever, Retriever
from buster.tokenizers import GPTTokenizer
from buster.validators import QuestionAnswerValidator, Validator
from buster.utils import extract_zip

from huggingface_hub import hf_hub_download


HUB_DB_FILE = "deeplake_store.zip"
REPO_ID = "jerpint/hf_buster_data"

hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=HUB_DB_FILE,
    local_dir=".",
)

extract_zip(zip_file_path=HUB_DB_FILE, output_path=".")


buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_response_templates": [
            "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        ],
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "invalid_question_response": "This question does not seem relevant to my current knowledge.",
        "check_question_prompt": """You are a chatbot answering technical questions on the huggingface documentation, a library used to train and do inference on open-source artificial intelligence models.

Your job is to determine wether or not a question is valid, and should be answered.
More general questions are not considered valid, even if you might know the response.
Questions that are likely to be related to the huggingface library are considered valid.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.

For example:

Q: How can I train a vision model?
true

Q: What is the meaning of life?
false

A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.""",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
    },
    retriever_cfg={
        "path": "deeplake_store",
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    documents_answerer_cfg={
        "no_documents_message": "No documents are available for this question.",
    },
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "formatter": "{content}",
    },
    prompt_formatter_cfg={
        "max_tokens": 3500,
        "text_before_docs": (
            "You are an chatbot answering technical questions on the huggingface transformers library. "
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documentation. "
            "If the answer is in the documentation, summarize it in a helpful way to the user. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "Here is the documentation: "
            "<DOCUMENTS> "
        ),
        "text_after_docs": (
            "<\DOCUMENTS>\n"
            "REMEMBER:\n"
            "You are an chatbot answering technical questions on the huggingface transformers library. "
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documentation above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not reference any links, urls or hyperlinks in your answers.\n"
            "4) Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "5) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?'"
            "For example:\n"
            "What is the meaning of life for an qa bot?\n"
            "I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with? "
            "Now answer the following question:\n"
        ),
    },
)


def setup_buster(buster_cfg: BusterConfig):
    """initialize buster with a buster_cfg class"""
    retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatter(
            tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg
        ),
        prompt_formatter=PromptFormatter(
            tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg
        ),
        **buster_cfg.documents_answerer_cfg,
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(
        retriever=retriever, document_answerer=document_answerer, validator=validator
    )
    return buster
