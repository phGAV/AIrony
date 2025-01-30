from fastapi import FastAPI, HTTPException
import logging
from meme_generator import create_meme_graph, MemeState, MemeRequest, MemeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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