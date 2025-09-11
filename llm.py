"""
Module for obtaining hints from OpenAI Chat Completions.
"""

import sys
from openai import OpenAI
from typing import Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Give a brief, confident suggestion "
    "(1-2 sentences), no fluff or apologies. Answer in English."
)


def make_hint(
    text: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Get a brief hint from the AI assistant.
    
    Args:
        text: Text to analyze
        client: OpenAI client
        model: Model to use
        temperature: Generation temperature
        system_prompt: Optional system message to steer assistant
    
    Returns:
        str: Assistant's response
    """
    if not text.strip():
        return ""
    
    try:
        # Use provided system prompt or default
        prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        
        # Send request
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        
        # Extract response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return ""
            
    except Exception as e:
        print(f"[error] Failed to get hint: {e}", file=sys.stderr)
        return ""
