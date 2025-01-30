import httpx
import random
import logging
import json
from fastapi import HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEME_TEMPLATES: List[Dict] = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Models
class MemeRequest(BaseModel):
    topic: str
    style: str = Field(default="funny", description="Meme style (funny, sarcastic, wholesome)")

class MemeResponse(BaseModel):
    url: str # TODO actually return 3 of 'em
    template_id: str
    text: List[str]

class TemplateInfo(BaseModel):
    template_id: str = Field(description="The unique identifier for the template")
    name: str = Field(description="The display name of the template")
    required_lines: int = Field(description="Number of text lines required for this template")
    example_usage: List[str] = Field(description="Example of how this template is typically used")

# class MemeGeneration(BaseModel):
#     template_id: str = Field(description="ID of the selected meme template, must be one of the provided template IDs")
#     text_lines: List[str] = Field(description="Text lines for the meme, number must match template's required_lines")

class MemeCandidate(BaseModel):
    template_id: str = Field(description="ID of the selected meme template, must be one of the provided template IDs")
    text_lines: List[str] = Field(description="Text lines for the meme, number must match template's required_lines")
    reasoning: str = Field(description="Explanation of why this meme works")
    humor_score: Optional[float] = None

class MemeCandidatesList(BaseModel):
    meme_candidates_list: List[MemeCandidate] = Field(description="The list of Meme Candidates")

class MemeBatch(BaseModel):
    candidates: List[MemeCandidate]
    humor_scores: List[float]
    selected_index: Optional[int] = None
    evaluation_notes: Optional[str] = None

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

async def create_meme(state: MemeState) -> MemeState:
    batch: MemeBatch = await generate_meme_candidates(state.topic, state.style, state.templates)
    logger.info(f'LLM Response: {batch}')

    # TODO try catch set state to error
    winner: MemeCandidate = batch.candidates[batch.selected_index]

    state.selected_template = next((t for t in state.templates if t['id'] == winner.template_id), None)
    state.meme_text = winner.text_lines
    return state

async def generate_meme_candidates(
    topic: str,
    style: str,
    templates: List[dict],
    num_candidates: int = 3
) -> MemeBatch:
    """Generate multiple meme candidates and select the best one"""
    
    template_info = [
        {
            "template_id": t['id'],
            "name": t['name'],
            "required_lines": t['lines'],
            "example_usage": t['example']['text']
        }
        for t in templates
    ]
    valid_template_ids = [t['id'] for t in templates]

    generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert meme creator who understands:
            - Meme culture and formats
            - Internet slang and references
            - Viral humor patterns
            - Template-specific conventions
            
            Generate multiple DISTINCT meme options, each:
            1. Using a different creative approach
            2. Choosing an appropriate template
            3. Generating matching text lines
            4. Including explanation of your creative reasoning"""),
        HumanMessagePromptTemplate.from_template("""
            Topic: {topic}
            Style: {style}
            Number of candidates: {num_candidates}

            Available Templates:
            {templates}
            
            For each candidate consider:
            - Template context and typical usage
            - Current internet trends
            - Relatable situations
            - Subversion of expectations
            - Memetic mutations of similar jokes
            
            For each of {num_candidates} candidates, provide:
            1. template_id (MUST be from valid IDs: {valid_ids})
            2. text_lines (MUST match template's required line count)
            3. reasoning (explain why this combination works and how it uses the considerations above)
            
            Make each candidate distinctly different in approach.""")
    ])

    evaluation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert meme critic with deep understanding of:
            - Internet culture and meme evolution
            - Humor theory and comedic timing
            - Template-specific meme conventions
            - Viral content patterns
            
            Your goal is to evaluate meme candidates objectively and select the one with highest potential impact."""),
        HumanMessagePromptTemplate.from_template("""
            Topic: {topic}
            Style: {style}
            
            Available Templates Context:
            {templates}
            
            Candidates:
            {candidates}
            
            For each candidate, evaluate:
            1. Memetic Potential (how likely to spread/be shared)
            2. Template Utilization (how well it uses the format)
            3. Humor Execution (timing, surprise, relatability)
            4. Style Match (alignment with requested {style} style)
            5. Cultural Relevance (connection to topic and current trends)
            
            Provide for each:
            1. Numerical score (0-10) for each aspect above
            2. Brief analysis of strengths/weaknesses
            3. Overall humor score (0-10)
            
            Then:
            1. Select the best candidate (provide index)
            2. Explain selection reasoning
            3. Suggest one small tweak that could improve the chosen meme further""")
    ])

    generator = ChatOpenAI(model='gpt-4o-mini', temperature=0.8).with_structured_output(MemeCandidatesList)
    generator_chain = generation_prompt | generator

    candidates: MemeCandidatesList = await generator_chain.ainvoke({
        "topic": topic,
        "style": style,
        "num_candidates": num_candidates,
        "templates": template_info,
        "valid_ids": valid_template_ids
    })

    evaluator = ChatOpenAI(model='gpt-4o-mini', temperature=0.3).with_structured_output(MemeBatch)
    evaluator_chain = evaluation_prompt | evaluator

    eval_response: MemeBatch = await evaluator_chain.ainvoke({
        "topic": topic,
        "style": style,
        "templates": template_info,
        "candidates": candidates.meme_candidates_list
    })
    logger.info(f'Evaluator response: {eval_response}')

    # TODO compare scores from generator and evaluator (hide those from generator first?)
    # for i, score in enumerate(eval_response["humor_scores"]):
    #     batch.candidates[i].humor_score = score

    return eval_response

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