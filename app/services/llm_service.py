from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class RetrievalObject:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=os.environ.get("googleAPIKey")
        )

    def createBaseResponseChain(self):
        system_template = """The context below consists of previous people speaking with their therapists about their issues, use these contexts 
as a pointer and then answer the query from the user based on the assistant's responses in the previous chats. Answer only questions relevant to mental health and don't answer if the question is irrelevant.
ABSOLUTELY AVOID ANSWERING IRRELEVANT QUESTIONS. NEVER USE THE NAME FROM THE CONTEXTS instead Ask the user for their name to make it feel more personal.

    Contexts:
    {context}"""

        system_message_template = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_template = "{input}"
        human_message_template = HumanMessagePromptTemplate.from_template(
            human_template
        )

        # Create prompt template
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_template,
                MessagesPlaceholder(variable_name="chat_history"),
                human_message_template,
            ]
        )

        # Create and return document chain
        return create_stuff_documents_chain(llm=self.llm, prompt=chat_prompt)

    def retrieverObject(self):
        self.retriever = self.createBaseResponseChain()
        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm, retriever=self.retriever, prompt=retriever_prompt
        )
        return history_aware_retriever

    def createChain(self, history_aware_retriever):
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            self.retriever,
        )
        return retrieval_chain
