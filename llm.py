"""
Модуль для получения подсказок от OpenAI Chat Completions.
"""

import sys
from openai import OpenAI
from typing import Optional


def make_hint(text: str, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """
    Получить краткую подсказку от AI ассистента.
    
    Args:
        text: Текст для анализа
        client: OpenAI клиент
        model: Модель для использования
        temperature: Температура генерации
    
    Returns:
        str: Ответ ассистента
    """
    if not text.strip():
        return ""
    
    try:
        # Системный промпт для кратких подсказок
        system_prompt = (
            "Ты - полезный AI ассистент. Дай краткий, уверенный совет "
            "(1-2 предложения), без воды и извинений. Отвечай на русском языке."
        )
        
        # Отправляем запрос
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        
        # Извлекаем ответ
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return ""
            
    except Exception as e:
        print(f"[error] Ошибка получения подсказки: {e}", file=sys.stderr)
        return ""
