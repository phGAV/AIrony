from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import httpx
import random
import logging
import json
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

MEME_TEMPLATES: List[Dict] = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

async def fetch_templates(state: MemeState) -> MemeState:
    global MEME_TEMPLATES
    
    if not MEME_TEMPLATES:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get("https://api.memegen.link/templates")
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to fetch templates")
                MEME_TEMPLATES = response.json()
                logger.info(f"Successfully loaded {len(MEME_TEMPLATES)} meme templates")
        except Exception as e:
            logger.exception("Error fetching templates")
            raise HTTPException(status_code=500, detail=str(e))
    state.templates = random.sample(MEME_TEMPLATES, min(10, len(MEME_TEMPLATES)))
    return state

class TemplateInfo(BaseModel):
    template_id: str = Field(description="The unique identifier for the template")
    name: str = Field(description="The display name of the template")
    required_lines: int = Field(description="Number of text lines required for this template")
    example_usage: List[str] = Field(description="Example of how this template is typically used")

class MemeGeneration(BaseModel):
    template_id: str = Field(description="ID of the selected meme template, must be one of the provided template IDs")
    text_lines: List[str] = Field(description="Text lines for the meme, number must match template's required_lines")

async def create_meme(state: MemeState) -> MemeState:
    valid_template_ids = [t['id'] for t in state.templates]
    template_info = [
        {
            "template_id": t['id'],
            "name": t['name'],
            "required_lines": t['lines'],
            "example_usage": t['example']['text']
        }
        for t in state.templates
    ]
    logger.info(f'Creating meme: made template list: {template_info}')

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(content="""You are an expert meme creator who understands:
                - Meme culture and formats
                - Internet slang and references
                - Viral humor patterns
                - Template-specific conventions
                You MUST: 
                1. Choose a template_id from the EXACTLY provided list
                2. Generate the exact number of text lines required by the chosen template
                3. Make the content match the requested style and topic"""),
            HumanMessagePromptTemplate.from_template("""
                Topic: {topic}
                Style: {style}
            
                Available Templates:
                {templates}
                Consider:
                - Template context and typical usage
                - Current internet trends
                - Relatable situations
                - Subversion of expectations
                - Memetic mutations of similar jokes
                         
                Rules:
                1. You MUST select a template_id from the above list
                2. You MUST provide exactly the number of text lines specified for the chosen template
                    
                Valid template IDs: {valid_ids}"""),
        ]
    )
    logger.info('Creating model Instance')
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7).with_structured_output(MemeGeneration)
    
    chain = prompt | model

    logger.info('Invoking chain')
    response: MemeGeneration = await chain.ainvoke({
        "topic": state.topic,
        "style": state.style,
        "templates": json.dumps(template_info, indent=2),
        "valid_ids": ", ".join(valid_template_ids)
    })

    logger.info(f'LLM Response: {response}')

    if response.template_id not in valid_template_ids:
        state.error = f"Selected template ID '{response.template_id}' not found in available templates"
        return state
    
    state.selected_template = next((t for t in state.templates if t['id'] == response.template_id), None)
    state.meme_text = response.text_lines
    return state

# Create Graph
def create_meme_graph() -> StateGraph:
    workflow = StateGraph(MemeState)
    
    workflow.add_node("fetch_templates", fetch_templates)
    workflow.add_node("create_meme", create_meme)
    
    workflow.set_entry_point("fetch_templates")
    workflow.add_edge("fetch_templates", "create_meme")
    workflow.add_edge("create_meme", END)
    
    logger.info('Made MemeState graph')
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
    logger.info(f"Received request: {request}")
    try:
        graph = create_meme_graph()
        
        initial_state = MemeState(
            topic=request.topic,
            style=request.style,
            templates=[]
        )
        logger.info("Invoking graph")
        final_dict = await graph.ainvoke(initial_state)
        final_state = MemeState.model_validate(final_dict)
        
        if final_state.error:
            raise HTTPException(status_code=400, detail=final_state.error)
        
        template_id = final_state.selected_template['id']
        lines_available = final_state.selected_template['lines']
        formatted_text = "/".join("".join(safe_chars.get(c, c) for c in line) for line in final_state.meme_text[:lines_available])

        # Generate meme URL
        meme_url = f"https://api.memegen.link/images/{template_id}/{formatted_text}.png"
        
        logger.info(f"Generated meme URL: {meme_url}")
        return MemeResponse(
            url=meme_url,
            template_id=template_id,
            text=final_state.meme_text
        )
    except Exception as e:
        logger.exception("Error in generate_meme endpoint")
        raise