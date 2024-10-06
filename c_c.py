import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from uvicorn import run

### Code
# Initialize the app
app = FastAPI()

# Define the model and the template
model = ChatOllama(model="qwen2.5:0.5b", temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}. Respond in 20 words or fewer"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

runnable = prompt | model
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Define input schema for asking a question
class QueryRequest(BaseModel):
    ability: str
    input: str
    session_id: str

# Define input schema for retrieving history
class HistoryRequest(BaseModel):
    session_id: str

# Define the FastAPI route to create a unique session ID (First step)
@app.post("/create-session")
async def create_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    return {"session_id": session_id}

# Define the FastAPI route to ask a question (Second step)
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Invoke the LLM with the session history
    response = with_message_history.invoke(
        {"ability": request.ability, "input": request.input},
        config={"configurable": {"session_id": request.session_id}},
    )
    return {"response": response.content}

# Define the FastAPI route to retrieve chat history (Third step)
@app.post("/history")
async def get_chat_history(request: HistoryRequest):
    # Get the session's chat history
    session_history = get_session_history(request.session_id)
    return {"history": session_history}

if __name__ == "__main__":
    run("c_c:app", host="localhost", port=8000, reload=True)
