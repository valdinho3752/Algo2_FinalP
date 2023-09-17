from fastapi import FastAPI
from pydantic import BaseModel
from Conversational_agent import Conv_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conv = Conv_agent()


class prompt(BaseModel):
    user_prompt : str


@app.post("/gpt/prompt")
async def Post_prompt(prompt : prompt):
    return {"response" : conv.conversation(prompt.user_prompt)}
