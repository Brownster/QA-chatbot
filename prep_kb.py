import os
from tkinter import *
from tkinter import ttk
from bs4 import BeautifulSoup
import requests
from pprint import pprint
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import TextSplitter
import tiktoken
from tkinter import messagebox


kb_db = None

os.environ["OPENAI_API_KEY"] = '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'

def extract_links(url, level=1, max_level=1):
    """
    Extracts all the links from the given URL using Beautiful Soup recursively up to max_level.
    Args:
        url (str): The URL to extract links from.
        level (int): The current level of recursion.
        max_level (int): The maximum level of recursion allowed.
    Returns:
        list[str]: A list of extracted links.
    """
    if level > max_level:
        return []

    try:
        response = requests.get(url)
        if response.status_code != 200:
            update_status(f"Error fetching or processing {url}, status code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        links = [link.get("href") for link in soup.find_all("a")]
        links = [link for link in links if link and link.startswith('http')]

        if level < max_level:
            child_links = []
            for link in links:
                child_links.extend(extract_links(link, level=level+1, max_level=max_level))
            links.extend(child_links)

        return links
    except Exception as e:
        update_status(f"Error extracting links from {url}, exception: {str(e)}")
        return []



def update_status(status_text):
    status_text_widget.config(state=NORMAL)
    status_text_widget.insert(END, status_text + "\n")
    status_text_widget.config(state=DISABLED)
    status_text_widget.yview(END)

def prepare_kb():
    global kb_db
    
    update_status("Extracting URLs from URLs provided...")
    start_urls = urls_text.get(1.0, END).strip().split("\n")
    all_urls = []

    # Hardcode the collection name and persist_directory
    persist_directory = os.path.join(os.getcwd(), "mykb_db")

    max_level = int(max_level_entry.get()) if max_level_entry.get().isdigit() else 3
    documents_directory = documents_directory_entry.get()

    update_status("Extracting links...")
    for url in start_urls:
        links = extract_links(url, max_level=max_level)
        all_urls.extend(links)
        update_status(f"Extracted links from {url}")

    unique_urls = list(set(all_urls))

    update_status("Loading documents from URLs...")
    loader = UnstructuredURLLoader(urls=unique_urls)
    
    try:
        kb_data = loader.load()
    except Exception as e:
        update_status(f"Error fetching or processing URL(s), exception: {str(e)}")
        return

    # Get the list of files in the directory
    if documents_directory and os.path.isdir(documents_directory):
        files = [os.path.join(documents_directory, f) for f in os.listdir(documents_directory) if os.path.isfile(os.path.join(documents_directory, f))]
    else:
        files = []

    # Loading local PDF files
    update_status("Processing local PDF files ...")
    pdf_data = []
    for file_path in files:
        if file_path.lower().endswith('.pdf'):
            loader = UnstructuredPDFLoader(file_path)
            try:
                document_data = loader.load()
                pdf_data.extend(document_data)
            except Exception as e:
                update_status(f"Error loading document {os.path.basename(file_path)}, exception: {str(e)}")

    # Loading local Word document files
    update_status("Processing local Word document files ...")
    doc_data = []
    for file_path in files:
        if file_path.lower().endswith('.docx'):
            try:
                loader = UnstructuredWordDocumentLoader(file_path)
                document_data = loader.load()
                doc_data.extend(document_data)
            except Exception as e:
                update_status(f"Error loading document {os.path.basename(file_path)}, exception: {str(e)}")

    kb_data.extend(pdf_data)
    kb_data.extend(doc_data)


    # Split the documents into smaller chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    kb_doc = text_splitter.split_documents(kb_data)

    # Generate embeddings for the documents
    embeddings = OpenAIEmbeddings()

    if kb_db is None:
        # Create a Chroma vector store from the documents and their embeddings
        kb_db = Chroma.from_documents(kb_doc,
                              embeddings,
                              persist_directory=persist_directory
                              )
    else:
        # Add new documents and their embeddings to the existing kb_db
        kb_db.add_documents(kb_doc, embeddings)

    # Persist the vector store
    update_status("Persisting the vector store...")
    kb_db.persist()
    update_status("Knowledge base preparation complete.")


def load_kb_db_path():
    try:
        with open('kb_db_path.txt', 'r') as file:
            return file.readline().strip()
    except FileNotFoundError:
        return None


def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """
    Returns the number of tokens used by a list of messages.
    Args:
    messages (list): A list of messages, each of which is a dictionary containing the role and content of the message.
    model (str): The name of the model to use for tokenization. Defaults to "gpt-4".
    Returns:
    int: The number of tokens used by the list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        # !Note: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # !Note: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows {role/name}\n{content}\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with assistant
    return num_tokens


def count_string_tokens(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.
    Args:
    string (str): The text string.
    model_name (str): The name of the encoding to use. (e.g., "gpt-4")
    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# truncate
def truncate_conversation_history(messages, max_tokens):
    while num_tokens_from_messages(messages) > max_tokens:
        messages.pop(0)  # Remove the oldest message
    return messages


def load_kb_db_path() -> str:
    kb_db_path = os.path.join(os.getcwd(), "mykb_db", "mykb_db.chroma")
    if os.path.isfile(kb_db_path):
        return kb_db_path
    else:
        return None
    

# ask a question
def ask_question():
    persist_directory = './mykb_db'
    kb_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    question = question_entry.get()
    if not question:
        messagebox.showwarning("Warning", "Please enter a question.")
        return

    if kb_db is None:
        kb_db_path_and_db = load_kb_db_path()
        if kb_db_path_and_db is not None:
            kb_db_path, kb_db = kb_db_path_and_db
            print(f"Loaded vector store from {kb_db_path}")
        else:
            messagebox.showwarning("Warning", "Please prepare the knowledge base before asking questions.")
            return

    query_statement = question_entry.get()
    chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    max_tokens_allowed = 4096 - 300  # Reserve some tokens for the response and buffer

    chat_history.append({"role": "user", "content": query_statement})
    truncated_messages = truncate_conversation_history(chat_history, max_tokens_allowed)

    kb_qa = ChatVectorDBChain.from_llm(
        OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        vectorstore=kb_db,
        top_k_docs_for_context=4,
        return_source_documents=True
    )

    result = kb_qa({"question": query_statement, "chat_history": truncated_messages})
    answer_text_widget.config(state=NORMAL)
    answer_text_widget.insert(END, "Question: " + query_statement + "\n")
    answer_text_widget.insert(END, "Answer: " + result["answer"] + "\n")
    answer_text_widget.insert(END, "Source Documents:\n")
    for doc in result["source_documents"]:
        answer_text_widget.insert(END, f"  - {doc['title']}\n")
    answer_text_widget.insert(END, "\n")
    answer_text_widget.config(state=DISABLED)
    answer_text_widget.yview(END)

if __name__ == "__main__":
    # Load the vector store at the beginning of the program
    load_kb_db_path()

# GUI setup
window = Tk()
window.title("Knowledge Base and Chatbot")

frame = Frame(window)
frame.pack(padx=20, pady=20)

# Create the tab control
tabControl = ttk.Notebook(frame)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)

tabControl.add(tab1, text="Prepare Knowledge Base")
tabControl.add(tab2, text="Chatbot")

tabControl.grid(row=0, column=0, columnspan=2)

# Create separate frames for each tab
tab1_frame = Frame(frame)
tab1_frame.grid(row=1, column=0, columnspan=2)

tab2_frame = Frame(frame)
tab2_frame.grid(row=1, column=0, columnspan=2)

# Move all the tab1 widgets to tab1_frame
urls_label = Label(tab1_frame, text="Enter URLs (one per line):")
urls_label.grid(row=0, column=0, sticky=NW)

urls_text = Text(tab1_frame, wrap=WORD, width=60, height=10)
urls_text.grid(row=0, column=1, pady=(0, 10), sticky=W)

max_level_label = Label(tab1_frame, text="Max Link Level:")
max_level_label.grid(row=1, column=0, sticky=W)

max_level_entry = Entry(tab1_frame, width=40)
max_level_entry.grid(row=1, column=1, sticky=W)

# Add a label and entry field for the documents directory
documents_directory_label = Label(tab1_frame, text="Documents Directory:")
documents_directory_label.grid(row=2, column=0, sticky=W)

documents_directory_entry = Entry(tab1_frame, width=40)
documents_directory_entry.grid(row=2, column=1, sticky=W)

prepare_button = Button(tab1_frame, text="Prepare", command=prepare_kb)
prepare_button.grid(row=3, column=1, pady=(10, 0), sticky=E)

status_label = Label(tab1_frame, text="")
status_label.grid(row=3, column=1, pady=(10, 0), sticky=W)

status_label = Label(tab1_frame, text="Status:")
status_label.grid(row=4, column=0, pady=(10, 0), sticky=W)

status_text_widget = Text(tab1_frame, wrap=WORD, width=60, height=10, state=DISABLED)
status_text_widget.grid(row=4, column=1, pady=(10, 0), sticky=W)

# Move all the tab2 widgets to tab2_frame
question_label = Label(tab2_frame, text="Enter your question:")
question_label.grid(row=0, column=0, sticky=W)

question_entry = Entry(tab2_frame, width=60)
question_entry.grid(row=0, column=1, sticky=W)

ask_button = Button(tab2_frame, text="Ask", command=ask_question)
ask_button.grid(row=1, column=1, pady=(10, 0), sticky=E)

answer_label = Label(tab2_frame, text="Answer and Source Documents:")
answer_label.grid(row=2, column=0, pady=(10, 0), sticky=W)

answer_text_widget = Text(tab2_frame, wrap=WORD, width=60, height=10, state=DISABLED)
answer_text_widget.grid(row=2, column=1, pady=(10, 0), sticky=W)

# switch between tabs
def switch_tabs(event, selected_tab_index):
    tab1_frame.grid_remove()
    tab2_frame.grid_remove()
    if selected_tab_index == 0:
        tab1_frame.grid(row=1, column=0, columnspan=2)
    elif selected_tab_index == 1:
        tab2_frame.grid(row=1, column=0, columnspan=2)


#Bind the tab change event to the function
def on_tab_changed(event):
    selected_tab = tabControl.index(tabControl.select())
    switch_tabs(event, selected_tab)

tabControl.bind("<<NotebookTabChanged>>", on_tab_changed)

#Start the GUI
window.mainloop()
