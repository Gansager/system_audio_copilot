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

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You are a facilitator. Summarize the meeting briefly and to the point, in the language of the meeting."
    " First 1-2 sentences: what the meeting was about (no fluff). Then a list of 3-8 key points: decisions, questions, next steps."
    " Don't add preambles like 'Here's a quick summary'. Use '-' markers for points."
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


def make_summary(
    transcript_text: str,
    client: OpenAI,
    model: str,
    temperature: float,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate a concise summary from transcript text.
    Returns empty string on failure.
    """
    if not transcript_text or not transcript_text.strip():
        return ""
    try:
        prompt = system_prompt if system_prompt else DEFAULT_SUMMARY_SYSTEM_PROMPT
        user_msg = transcript_text
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        return ""
    except Exception as e:
        print(f"[warning] Summary generation failed: {e}", file=sys.stderr)
        return ""
