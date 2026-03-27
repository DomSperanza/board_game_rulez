"""
Module: gemini_client
Purpose:
- Initialize the Google Generative AI (Gemini) client.
- Build the final prompt by combining the relevant chunks (from search.py) and the user's question.
- Call the Gemini free tier API to generate a helpful, concise answer.
"""
import os
import google.generativeai as genai

# Configure API key (Assuming it's loaded in the environment)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

def get_answer(query: str, context_chunks: list[str]) -> str:
    """Gets an answer from Gemini based on the provided context."""
    if not context_chunks:
        return "I couldn't find any relevant rules for that question in the rulebook."
        
    # Build prompt
    context_text = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
You are a helpful board game rules expert.
Answer using ONLY the official rulebook excerpts below.
If the excerpts list resources, quantities, or a "Requires:" line, include those details exactly.
If the answer is not in the excerpts, say you cannot find it in the rulebook. Be concise.

RULEBOOK EXCERPTS:
{context_text}

USER QUESTION:
{query}
"""

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]
    
    last_error = ""
    for specific_model in models_to_try:
        try:
            model = genai.GenerativeModel(specific_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            continue
            
    print(f"Error calling Gemini: {last_error}")
    return "Sorry, I encountered an error connecting to the API. Check terminal logs."
