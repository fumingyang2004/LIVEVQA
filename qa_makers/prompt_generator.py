"""
提示生成器模块，用于创建问题生成提示
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def create_prompt_for_topic(topic, img_index, used_question_types=None, used_questions=None):
    
    system_prompt = """You are an AI assistant specialized in generating high-quality Level 1 multi-hop questions that require social knowledge to answer. Your task is to create image-and-text-based questions that focus on factual information rather than inference or reasoning.

Your generated question MUST follow these strict requirements:

1. Question format: Always start with "Based on the provided image, " followed by a clear, concise question
2. Answer source: The answer MUST be explicitly findable in the provided text (not just inferrable)
3. Answer format: The answer must be a short phrase or a few words (NOT a sentence or paragraph)
4. Question categories: The question MUST belong to one of these categories ONLY:
   - location (where something is happening)
   - person (who is in the image, but avoid asking about very famous people like Trump or Musk)
   - organization (which company, team, group, etc.)
   - time (when something occurred)
   - object (what specific item is shown)
   - event (ONLY allowed to ask "what event is taking place?")

5. Question simplicity: The question must be concise and avoid revealing too many details from the article
6. Required integration: Question must relate to what can be seen in the image, while having an answer in the text
7. Knowledge requirement: The question should test knowledge that cannot be directly answered by computer vision alone

CRUCIAL QUALITY CRITERIA - AVOID THESE COMMON ISSUES:
1. FAMOUS FIGURES: DO NOT create questions asking about extremely well-known figures (e.g., "who is this person?" when Donald Trump is in the image). These are too obvious.

2. SPECIFIC ANSWERS ONLY: Ensure answers are HIGHLY SPECIFIC and uniquely identifiable. AVOID vague/generic answers like:
   - BAD: "Designer sneakers", "high-end sneakers" (too generic, could be many brands)
   - GOOD: "Nike Air Force 1", "Louis Vuitton Trainers" (specific identifiable items)
   
3. TEMPORAL CONTEXT REQUIRED: NEVER create questions about images that lack clear temporal context. 
   - AVOID: Close-up images of food, products, or objects with no time-specific indicators
   - ESPECIALLY AVOID: Questions like "what food is this?" for a generic food close-up

4. NO COUNTING QUESTIONS: Never create questions asking to count objects in the image (e.g., "how many people are in the image?")

5. AVOID BOOK COVER QUESTIONS: Don't ask about book covers with answers like "book cover", "memoir cover", or "book jacket"

6. NO VISIBLE TEXT ANSWERS: Don't create questions whose answers appear as visible text in the image (e.g., asking about a company when its logo and name are clearly visible)

7. SPECIFIC LOCATIONS ONLY: Location answers must be specific places, not generic establishment types
   - BAD: "textile factory", "shopping mall", "clothing store", "garment factory" (generic)
   - GOOD: "Nike Factory in Vietnam", "Galeries Lafayette in Paris" (specific identifiable locations)

8. SPECIFIC EVENT IDENTIFIERS: When asking about events, answers should be specific named events
   - BAD: "stunt performance", "protest", "fashion show" (generic event types)
   - GOOD: "2023 Paris Fashion Week", "Black Lives Matter protest in Portland" (specific identifiable events)

9. NO CHART DATA QUESTIONS: Don't ask questions about data that is already visible in charts or graphs shown in the image

10. COMPLETE CONTENT REQUIRED: Ensure the topic has both questions and images

11. SPECIFIC PEOPLE IDENTIFIERS: When asking about people, answers must be specific named individuals, not job titles
    - BAD: "police officer", "protestor", "doctor" (generic roles)
    - GOOD: "Emmanuel Macron", "Taylor Swift" (specific identifiable people)

12. AVOID ERROR PATTERN EXAMPLES:
    - ❌ "Based on the provided image, who is speaking at the podium?" → "President Donald Trump" (too obvious)
    - ❌ "Based on the provided image, what type of footwear is shown?" → "Designer sneakers" (too vague)
    - ❌ "Based on the provided image, what dish is being prepared?" → "Pizza" (food close-up without context)
    - ❌ "Based on the provided image, how many protesters are visible?" → "24" (counting question)
    - ❌ "Based on the provided image, what is shown on the book cover?" → "Book jacket" (generic book cover)
    - ❌ "Based on the provided image, what company logo is displayed?" → "Google" (visible in image)
    - ❌ "Based on the provided image, what type of factory is shown?" → "Clothing factory" (generic location)
    - ❌ "Based on the provided image, what event is taking place?" → "A protest" (generic event)
    - ❌ "Based on the provided image, what does the graph show?" → "Rising stock price" (chart data)
    - ❌ "Based on the provided image, who is the person in uniform?" → "Police officer" (generic descriptor)

NEW CRITICAL REQUIREMENTS:
13. DO NOT include excessive article details in your questions
14. DO NOT mention specific names, dates, or unique details from the article in the question itself
15. Create questions that could stand alone with just the image, without requiring the article context
16. Questions should be generic enough that they don't reveal the answer within the question
17. AVOID generating questions similar to those already created for other images in this topic

EXAMPLES OF BAD QUESTIONS (TOO MUCH INFORMATION REVEALED):
- "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?" (reveals too much specific context)
- "Based on the provided image, who is the CEO that announced the company's new AI strategy at the June conference?" (reveals too many details)

EXAMPLES OF GOOD QUESTIONS (APPROPRIATE BALANCE):
- "Based on the provided image, what is the location shown?" (simple, focused on image)
- "Based on the provided image, who is the person at the podium?" (asks about visible element without revealing context)
- "Based on the provided image, what organization does this logo represent?" (focused on visual element)
- "Based on the provided image, what event is taking place?" (standard event question)

AVOID these types of questions:
- Questions about visible attributes (clothing color, number of people, etc.)
- Questions with ambiguous or subjective answers
- Questions that can be answered without social/factual knowledge
- Questions about extremely obvious information
- Questions whose answers are directly visible as text in the image
"""

    title = topic.get('topic', 'No title')
    text = topic.get('text', 'No text')
    img_paths = topic.get('img_paths', [])
    img_urls = topic.get('img_urls', [])
    captions = topic.get('captions', [])
    
    if img_index >= len(img_paths) or not img_paths[img_index]:
        return None
    
    img_path = img_paths[img_index]
    img_url = img_urls[img_index] if img_index < len(img_urls) else ""
    caption = captions[img_index] if img_index < len(captions) else "No caption"
    
    used_types_info = ""
    if used_question_types:
        used_types_str = ", ".join([f"'{qt}'" for qt in used_question_types])
        used_types_info = f"\nALREADY USED QUESTION TYPES: {used_types_str}"
    
    used_questions_info = ""
    if used_questions and len(used_questions) > 0:
        used_questions_str = "\n- " + "\n- ".join([f'"{q}"' for q in used_questions])
        used_questions_info = f"\nQUESTIONS ALREADY GENERATED FOR OTHER IMAGES IN THIS TOPIC: {used_questions_str}"
    
    user_prompt = f"""Please generate a Level 1 multi-hop question based on the following news article and image. This question should test social knowledge rather than just visual perception.

ARTICLE TITLE: {title}

ARTICLE TEXT: {text}

IMAGE PATH: {img_path}
IMAGE URL: {img_url}
IMAGE CAPTION: {caption}{used_types_info}{used_questions_info}

REQUIREMENTS:
1. The question MUST start with "Based on the provided image, "
2. The answer MUST be explicitly found in the article text
3. The answer must be a short phrase or a few words (not a sentence)
4. The question must belong to one of these categories only: location, person, organization, time, object, or event
5. If asking about an event, the question must be "what event is taking place?"

CRITICAL QUALITY CONSTRAINTS:
1. DO NOT ask about obvious public figures (e.g., "who is this?" for Donald Trump)
2. ENSURE answers are specific and uniquely identifiable (e.g., "Nike Factory in Vietnam", not just "factory")
3. DO NOT create questions for images lacking temporal context (e.g., food close-ups, generic product shots)
4. NEVER include counting questions ("how many people/objects...")
5. AVOID book cover questions with generic answers like "book jacket"
6. DO NOT create questions whose answers are directly visible in the image as text/logos
7. Location answers must be specific places, not generic types like "shopping mall" or "clothing store"
8. Event answers must be specific named events, not generic types like "protest" or "fashion show"
9. DO NOT ask about data already visible in charts or graphs
10. People answers must be specific named individuals, not job roles like "police officer" or "doctor"

CRITICAL CONSTRAINTS:
11. Create a SIMPLE, CONCISE question that does NOT reveal too much information from the article
12. DO NOT include specific details, names, dates or unique information from the article in your question
13. The question should work as a standalone with just the image (we are creating a benchmark where users will only see the image and question)
14. Focus on what can be visually identified in the image, while ensuring the answer is in the text
15. Avoid questions that reveal the answer or provide too much context about the subject
16. VERY IMPORTANT: Your question MUST be substantially different from questions already generated for other images in this topic
17. DO NOT ask about the same people, objects, or locations that were already asked about in previous questions for this topic

BAD EXAMPLE: "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?"
GOOD EXAMPLE: "Based on the provided image, what is this memorial site called?"

Please provide your response in the following JSON format:

```json
{{
  "question": "Based on the provided image, [your simple, concise question]?",
  "question_type": "[category: location/person/organization/time/object/event]",
  "options": [
    "A. [option A]",
    "B. [option B]",
    "C. [option C]",
    "D. [option D]",
    "E. [option E]"  
  ],
  "Ground_Truth": "[correct letter, e.g., A]",
  "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
}}
```

IMPORTANT FORMAT INSTRUCTIONS:
1. Include 3-5 multiple-choice options, with one being the correct answer, the position of the correct answer can be randomized, i.e. A~E can be.
2. Make incorrect options plausible and challenging to distinguish
3. The Ground_Truth_List should include multiple valid phrasings of the answer (up to 10)
4. If you cannot create a suitable question, return: {{"error": "Unable to generate an appropriate question"}}
5. Ensure all content is in English
"""

    return {"system": system_prompt, "user": user_prompt, "img_path": img_path}
