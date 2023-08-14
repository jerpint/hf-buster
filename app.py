import cfg
import gradio as gr
import pandas as pd
from cfg import setup_buster

buster = setup_buster(cfg.buster_cfg)


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    matched_documents.similarity_to_answer = (
        matched_documents.similarity_to_answer * 100
    )

    # print the page instead of the heading, more meaningful for hf docs
    matched_documents["page"] = matched_documents.apply(
        lambda x: x.url.split("/")[-1], axis=1
    )

    documents_answer_template: str = "📝 Here are the sources I used to answer your question:\n\n{documents}\n\n{footnote}"
    document_template: str = "[🔗 {document.page}]({document.url}), relevance: {document.similarity_to_answer:2.1f} %"

    documents = "\n".join(
        [
            document_template.format(document=document)
            for _, document in matched_documents.iterrows()
        ]
    )
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return documents_answer_template.format(documents=documents, footnote=footnote)


def add_sources(history, completion):
    if completion.answer_relevant:
        formatted_sources = format_sources(completion.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    completion = buster.process_input(user_input)

    history[-1][1] = ""

    for token in completion.answer_generator:
        history[-1][1] += token

        yield history, completion


block = gr.Blocks()

with block:
    gr.Markdown(
        """<h1><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h1>"""
    )
    gr.Markdown(
        """
    #### This chatbot is designed to answer any questions related to the [huggingface transformers](https://huggingface.co/docs/transformers/index) library.
    #### It uses ChatGPT + embeddings to search the docs for relevant sections and uses them to answer questions. It can then cite its sources back to you to verify the information.
    #### Note that LLMs are prone to hallucination, so all outputs should always be vetted by users.

    #### The Code is open-sourced and available on [Github](www.github.com/jerpint/buster)")
    """
    )

    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            question = gr.Textbox(
                label="What's your question?",
                placeholder="Ask a question to AI stackoverflow here...",
                lines=1,
            )
        submit = gr.Button(value="Send", variant="secondary")

    examples = gr.Examples(
        examples=[
            "What kind of models should I use for images and text?",
            "When should I finetune a model vs. training it form scratch?",
            "Can you give me some python code to quickly finetune a model on my sentiment analysis dataset?",
        ],
        inputs=question,
    )

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand.")

    response = gr.State()

    submit.click(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])
    question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])


block.queue(concurrency_count=16)
block.launch(debug=True, share=False)
