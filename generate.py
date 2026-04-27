# LLM generation step for RAG pipeline
# using GPT-4o for better accuracy on banking docs
# TODO: add streaming response later

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

def build_prompt():
    template = """
    Use the following context to answer the question.
    If you don't know the answer say I don't know.
    Don't make up answers.

    Context: {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

def generate_answer(prompt, context, question):
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0
        # temperature=0 gives more consistent answers
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return response.content
