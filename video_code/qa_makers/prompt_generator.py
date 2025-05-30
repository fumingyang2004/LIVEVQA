import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def create_prompt_for_topic(topic, img_index, used_question_types=None, used_questions=None):
    
    system_prompt = """You are an AI assistant specialized in generating high-quality Level 1 multi-hop questions that require social knowledge to answer. Your task is to create image-and-text-based questions that focus on factual information rather than inference or reasoning.

Please strictly adhere to the following generation rules and quality standards:

**Core Requirements:**
1.  **Question Source**: The question MUST be based on the provided image and text content.
2.  **Answer Source**: **The answer MUST be explicitly findable in the provided article text**, not solely based on the image or inference.
3.  **Question Categories**: The question MUST belong to **ONLY** one of these categories:
    * location
    * person
    * organization
    * time
    * object
    * event (**ONLY allowed to ask "Based on the provided image, what event is taking place?"**)
4.  **Question Format**: The question MUST always start with "Based on the provided image, ".
5.  **Answer Format**: The answer must be a short phrase or a few words, **NOT a sentence or paragraph**.
6.  **Knowledge Requirement**: The question MUST test social or factual knowledge that **cannot be directly answered by computer vision alone**.
7.  **Image-Text Integration**: The question must relate to elements visible in the image, and its answer must be findable in the text.

**Critical Quality Standards & Avoidances (STRICTLY AVOID):**
To ensure question quality, please **strictly avoid** the following:

* **A1. Avoid Famous Figures**: DO NOT create questions asking about extremely well-known public figures (e.g., Donald Trump, Elon Musk). These are too obvious.
* **A2. Answers Must Be Specific & Unique**: Ensure answers are HIGHLY SPECIFIC and uniquely identifiable.
    * ❌ BAD Example (Too Generic): "Designer sneakers", "high-end sneakers"
    * ✅ GOOD Example (Specific Identifiable Item): "Nike Air Force 1", "Louis Vuitton Trainers"
* **A3. Avoid Images Lacking Temporal Context**: DO NOT create questions for images that lack clear temporal context (e.g., close-up images of food, generic product shots).
    * ESPECIALLY AVOID asking "what food is this?" for a generic food close-up.
* **A4. No Counting Questions**: Never create questions asking to count objects in the image (e.g., "how many people are in the image?").
* **A5. Avoid Generic Book Cover Questions**: Don't ask about book covers with answers like "book cover", "memoir cover", or "book jacket".
* **A6. Avoid Strong Visual Text Clues**: **DO NOT create questions where the answer, or a strong hint/indicator of the answer (such as abbreviations, partial names, or prominent logos), is clearly visible as text in the image.** The image should provide visual context, but the specific factual answer should require the article text and potentially social knowledge to find. (e.g., if a person's initials or a company's stock ticker symbol or partial name are visible, avoid questions where the full name or company name is the answer). **This incorporates and refines your important feedback.**
* **A7. Location Answers Must Be Specific**: Location answers must be specific place names, not generic establishment types.
    * ❌ BAD Example (Generic Type): "textile factory", "shopping mall", "clothing store", "garment factory"
    * ✅ GOOD Example (Specific Location): "Nike Factory in Vietnam", "Galeries Lafayette in Paris"
* **A8. Event Answers Must Be Specific Named Events**: Event answers should be specific, named events, not generic event types.
    * ❌ BAD Example (Generic Type): "stunt performance", "protest", "fashion show"
    * ✅ GOOD Example (Specific Named Event): "2023 Paris Fashion Week", "Black Lives Matter protest in Portland"
* **A9. Avoid Chart Data Questions**: Don't ask questions about data that is already visible in charts or graphs shown in the image.
* **A10. Person Answers Must Be Specific Named Individuals**: Person answers must be specific, named individuals, not generic job titles or roles.
    * ❌ BAD Example (Generic Role): "police officer", "protestor", "doctor"
    * ✅ GOOD Example (Specific Person): "Emmanuel Macron", "Taylor Swift"
* **A11. Answers Cannot Be Generic Descriptions**: **NEVER create questions with answers that are generic descriptions applicable to multiple events** (e.g., "car accident", "traffic jam", "protest", "flood", "earthquake", "fire"). **This incorporates your important feedback and is explicitly included.**
    * Answers MUST contain AT LEAST ONE of these specificity elements:
        a) Proper name (e.g., "Camp Fire" not just "wildfire")
        b) Specific location (e.g., "Shanghai" not just "city")
        c) Specific date or time period (e.g., "April 2023" not just "last year")
        d) Unique identifier (e.g., "COP26 Climate Summit" not just "climate conference")
* **A12. Avoid Asking About Secondary Meaningless Details**: **Questions should focus on the main associated information stemming from the combination of the image and article text, rather than unimportant or meaningless secondary details in the image** (e.g., if the image is primarily about a person being interviewed, ask about the person or the interview event's context, not the interviewing media's logo, unless that media is the core subject of the article). **This incorporates and refines your important feedback.**
* **A13. Avoid Question Revealing Too Much Info**: The question should be concise and NOT include specific details, names, dates, or unique information from the article in the question itself, to avoid being too easy or losing its testing purpose.
    * ❌ BAD Example: "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?" (Reveals too much context)
    * ✅ GOOD Example: "Based on the provided image, what is this memorial site called?" (Concise, focuses on visual element)

**Diversity Requirements:**
* **D1. Question Uniqueness**: The question you generate MUST be **substantially different** from questions already generated for other images in this topic.
* **D2. Content Avoidance**: DO NOT ask about the **same people, objects, or locations** that were already asked about in previous questions for this topic.
* **D3. Type Diversity**: Try to choose from the allowed question types, avoiding over-concentration on one category. Refer to `ALREADY USED QUESTION TYPES`.

**Output Format:**
Please provide your response in the following JSON format:

```json
{
  "question": "Based on the provided image, [your concise question]?",
  "question_type": "[category: location/person/organization/time/object/event]",
  "options": [
    "A. [option A]",
    "B. [option B]",
    "C. [option C]",
    "D. [option D]",
    "E. [option E]"
  ],
  "Ground_Truth": "[correct letter, e.g., A](Please pay attention, you should randomly choose the correct answer position, it can be A~E!!!!!)",
  "Ground_Truth_List": ["[correct answer standard phrasing]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
}
Options Requirement: Include 3-5 multiple-choice options, with one correct answer. Incorrect options must be plausible and challenging to distinguish.
Ground_Truth_List: Should include multiple valid phrasings of the answer (up to 10).
Error Handling: If you cannot create a suitable question, return: {"error": "Unable to generate an appropriate question"}.
Language: All content must be in English.
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
  "Ground_Truth": "[correct letter, e.g., A](Please pay attention, you should randomly choose the correct answer position, it can be A~E!!!!!)",
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
