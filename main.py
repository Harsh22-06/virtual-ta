from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx
import json
import logging
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

with open("scraped_data.json", "r", encoding="utf-8") as f:
    SCRAPED_DATA = json.load(f)

AI_PIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDIzMDFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ITIOceD3QhhMCP3IB3mZoJI5zFKnlhoXtVFgPrk-d_E"
AI_PIPE_API_URL = "https://aipipe.org/openrouter/v1/chat/completions"

# Define the response model
class Link(BaseModel):
    url: str
    text: str

class Response(BaseModel):
    answer: str
    links:List[Link]

class Item(BaseModel):
    question: str
    image: Optional[str] = None

def create_link(url: str, title: str = None) -> Link:
    """Helper function to create Link objects with proper error handling"""
    try:
        # Clean and validate URL
        clean_url = url.strip().strip('"')
        if not clean_url.startswith('http'):
            logger.error(f"Invalid URL format: {url}")
            return None
            
        # Clean and validate title
        clean_title = title.strip() if title else None
        if not clean_title:
            logger.warning(f"Empty title for URL: {url}")
            # Extract domain name as fallback title
            domain = urlparse(clean_url).netloc
            clean_title = f"Link from {domain}"
            
        logger.info(f"Creating link - URL: {clean_url}, Title: {clean_title}")
        return Link(url=clean_url, text=clean_title)
        
    except Exception as e:
        logger.error(f"Error creating link object - URL: {url}, Title: {title}, Error: {str(e)}")
        return None

@app.post("/submit", response_model=Response)
async def submit_string(item: Item):
    try:
        links = []
        context = ""
        for topic, content in SCRAPED_DATA.items():
            try:
                is_discourse = topic.startswith("discourse/")
                
                if is_discourse:
                    title = content.get("title", "")
                    body = content.get("content", "")
                    if item.question.lower() in title.lower() or item.question.lower() in body.lower():
                        context = context + f"Title of discourse topic: {title}\n\nContent: {body}"[:2000]
                        url = content.get("url", "").strip('"')
                        link = create_link(url=url, title=title)
                        if link:
                            links.append(link)
                        break
                else:
                    # Remove '#/' prefix from topic for comparison
                    clean_topic = topic[2:] if topic.startswith('#/') else topic
                    # Look for topic keywords in the question instead
                    if any(word.lower() in item.question.lower() for word in clean_topic.split()) or item.question.lower() in content.lower():
                        context = context+content[:2000]
                        url = f"https://tds.s-anand.net/{topic}"
                        lines = content.split('\n')
                        for line in lines:
                            if 'http' in line and ('discourse.onlinedegree.iitm.ac.in' in line or 'study.iitm.ac.in' in line):
                                url = line[line.find('http'):].split()[0].strip().strip('"')
                                
                                # Extract topic ID from URL
                                parts = url.split('/')
                                topic_id = None
                                for i in range(len(parts)):
                                    if parts[i] == "t" and i + 2 < len(parts):
                                        topic_id = parts[i+2]
                                        break
                                
                                # For regular topics with discourse links
                                if topic_id:
                                    discourse_key = f"discourse/{topic_id}"
                                    if discourse_key in SCRAPED_DATA:
                                        title = SCRAPED_DATA[discourse_key]["title"]
                                        link = create_link(url=url, title=title)
                                        if link:
                                            links.append(link)
                                    else:
                                        logger.warning(f"Discourse topic {topic_id} not found in SCRAPED_DATA")
                                        link = create_link(url=url)  # Will use domain name as fallback title
                                        if link:
                                            links.append(link)
                                break
            except Exception as e:
                logger.error(f"Error processing topic {topic}: {str(e)}")
                continue

        # Process matched content
        try:
            headers = {"Authorization": f"Bearer {AI_PIPE_TOKEN}"}
            payload = {
                "model": "google/gemini-2.0-flash-lite-001",
                "messages": [
                    {
                        "role": "user", 
                        "content": f"""Answer this question in 2-3 sentences using Markdown: '{item.question}'
                        Use information from this context. If the context doesn't contain relevant information, 
                        provide a brief, practical answer.
                        
                        Context: {context}"""
                    } 
                ]
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(AI_PIPE_API_URL, json=payload, headers=headers)
                response.raise_for_status()  # Raise error for bad status codes
                data = response.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer received.")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            raise HTTPException(status_code=503, detail="AI service unavailable")
        except Exception as e:
            logger.error(f"Error calling AI service: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

        return Response(
            answer=answer.strip(),
            links=links
        )
    except Exception as e:
        logger.error(f"Unhandled  error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
