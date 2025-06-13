# VIRTUAL TA (TDS)
## FastAPI Q&A System with AI Integration

### Description
A FastAPI-based application that processes questions using context from both course materials and discourse topics. It leverages the Gemini AI model to generate contextual answers and provides relevant source links.

### Features

* Question processing with context matching
* Integration with Gemini AI model
* Discourse topic integration
* Smart link generation
* Markdown-formatted responses
* Error handling and logging

### Project Structure
* `main.py` - FastAPI application
* `scrape.py` - Discourse scraping utilities
* `scraped_data.json` - Data storage
* `requirements.txt` - Dependencies
  
### API Usage
Send POST request to `/submit`:
```json
{
    "question": "Your question here",
    "image": "Optional image URL"
}
```