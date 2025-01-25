from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Models
class MemeRequest(BaseModel):
    topic: str
    style: str = Field(default="funny", description="Meme style (funny, sarcastic, wholesome)")

class MemeResponse(BaseModel):
    url: str
    template_id: str
    text: List[str]

# LangGraph State Type
class MemeState(BaseModel):
    topic: str
    style: str
    templates: List[Dict]
    selected_template: Optional[Dict] = None
    meme_text: Optional[List[str]] = None
    error: Optional[str] = None

# Node Functions
async def fetch_templates(state: MemeState) -> MemeState:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.memegen.link/templates")
        state.templates = response.json()[:10]  # Limit templates for prompt size
    return state

class MemeGeneration(BaseModel):
    template_id: str = Field(description="ID of the selected meme template")
    top_text: str = Field(description="Text to display at the top of the meme")
    bottom_text: str = Field(description="Text to display at the bottom of the meme")

async def create_meme(state: MemeState) -> MemeState:
    valid_template_ids = [t['id'] for t in state.templates]
    templates_list = "\n".join(f"- {t['id']}: {t['name']}" for t in state.templates)
    
    meme_tool = Tool(
        name="create_meme",
        description="Generate a meme using the selected template and text",
        args_schema=MemeGeneration,
        func=lambda x: x  # Dummy function as we just need the schema
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative meme generator. Select the most appropriate template and generate witty text for it.
        Use the create_meme tool to generate the final meme."""),
        ("user", """Topic: {topic}
        Style: {style}
        
        Available templates (ID: Description):
        {templates_list}""")
    ])
    
    model = ChatOpenAI(model='gpt-4', temperature=0.7).bind(
        functions=[convert_to_openai_function(meme_tool)],
        function_call={"name": "create_meme"}
    )
    
    chain = prompt | model
    
    response = await chain.ainvoke({
        "topic": state.topic,
        "style": state.style,
        "templates_list": templates_list
    })
    
    # Extract function call arguments
    result = response.additional_kwargs["function_call"]["arguments"]
    meme_params = JsonOutputParser().parse(result)
    
    # Validate template_id exists
    if meme_params["template_id"] not in valid_template_ids:
        state.error = f"Selected template ID '{meme_params['template_id']}' not found in available templates"
        return state
    
    state.selected_template = next(t for t in state.templates if t['id'] == meme_params["template_id"])
    state.meme_text = [meme_params["top_text"], meme_params["bottom_text"]]
    return state

# Create Graph
def create_meme_graph() -> StateGraph:
    workflow = StateGraph(MemeState)
    
    workflow.add_node("fetch_templates", fetch_templates)
    workflow.add_node("create_meme", create_meme)
    
    workflow.set_entry_point("fetch_templates")
    workflow.add_edge("fetch_templates", "create_meme")
    workflow.add_edge("create_meme", END)
    
    return workflow.compile()

safe_chars = {
    " ": "_",
    "_": "__",
    "-": "--",
    "\n": "~n",
    "?": "~q",
    "&": "~a",
    "%": "~p",
    "#": "~h",
    "/": "~s",
    "\\": "~b",
    "<": "~l",
    ">": "~g",
    "\"": "''",
}

# FastAPI Endpoints
@app.post("/generate_meme", response_model=MemeResponse)
async def generate_meme(request: MemeRequest):
    graph = create_meme_graph()
    
    initial_state = MemeState(
        topic=request.topic,
        style=request.style,
        templates=[]
    )
    
    final_dict = await graph.ainvoke(initial_state)
    final_state = MemeState.model_validate(final_dict)
    
    if final_state.error:
        raise HTTPException(status_code=400, detail=final_state.error)
    
    template_id = final_state.selected_template['id']
    text_top = "".join(safe_chars.get(c, c) for c in final_state.meme_text[0])
    text_bottom = "".join(safe_chars.get(c, c) for c in final_state.meme_text[1])

    # Generate meme URL
    meme_url = f"https://api.memegen.link/images/{template_id}/{text_top}/{text_bottom}.png"
    
    return MemeResponse(
        url=meme_url,
        template_id=template_id,
        text=final_state.meme_text
    )