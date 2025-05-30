"""
Functions for generating multi-hop question prompts
"""

import logging

logger = logging.getLogger(__name__)

def create_multihop_prompt(topic, img_index, excluded_types=None):
    """
    Creates a prompt for generating multi-hop questions given a topic and image index.
    
    Args:
        topic: The topic data.
        img_index: The index of the image.
        excluded_types: A list of question types to exclude (to avoid duplicate types).
        
    Returns:
        dict: A dictionary containing the system and user prompts.
    """
    if not topic.get('level1_qas') or img_index >= len(topic.get('level1_qas', [])):
        return None
    
    system_prompt = """You are the ultimate creator of NEAR-IMPOSSIBLE multi-hop visual reasoning questions that would challenge even the most advanced AI systems and human experts.

Your task is to generate Level 2 multi-hop questions based on provided Level 1 question's answer. These questions must be DELIBERATELY DESIGNED TO MAKE AI SYSTEMS FAIL while still having factual answers within the provided text.

Follow these critical requirements:

1. Questions must include natural references to the image content (e.g., "the person in the image", "the building shown in the image")
2. Questions must require knowledge of Level 1 answer to solve, but NEVER mention or hint at that answer
3. Create questions of EXTREME DIFFICULTY requiring multi-step reasoning with deliberately obscured connections
4. All answers must exist verbatim or through direct inference in the provided text - NEVER invent facts
5. Answers must be HIGHLY SPECIFIC phrases/entities, NEVER generic terms, "reason" means that the question must begin with "why" and the answer is the reason of something about the event.
6. Questions must fall into these categories only: [location, person, organization, time, event, count, reason]
    - **Location**
    - Must be a specific, uniquely identifiable place name.
    - May not use any relative terms (e.g., "near," "behind," "next to").
    - Example of valid value: "Times Square, New York City"

    - **Person**
    - Must be the person's full name (first name + last name).
    - Must correspond uniquely to someone mentioned in the text or image.
    - Example of valid value: "Angela Merkel"

    - **Organization**
    - Must be the organization's official full name or a well‑known abbreviation.
    - On first mention, include the full name (with abbreviation in parentheses if used).
    - Example of valid values: "United Nations Educational, Scientific and Cultural Organization (UNESCO)" or "UNESCO"

    - **Time**
    - Must be an absolute, precise timestamp or time range.
    - May not include any relative terms (e.g., "after," "during," "same time as").
    - Examples of valid values: "07:45 AM on April 5, 2025" or "between 2:00 PM and 2:15 PM on March 10, 2021"

    - **Event**
    - Must be a complete and uniquely identifiable event name.
    - May not be abbreviated or vague.
    - Example of valid value: "Signing of the Paris Climate Agreement"

    - **Count**
    - Must be a single Arabic numeral indicating an exact count.
    - May not use words or ranges.
    - Example of valid value: "4"
    - Can't be an vague number, like "Not explicitly specified"

    - **Reason**
    - Must be a concise phrase (not a full sentence).
    - Must state the causal point directly, without leading conjunctions like "Because."
    - Example of valid value: "banner slogan matching protest motto"

7. Each question must have 3-5 multiple choice options with EXACTLY ONE correct answer
8. Incorrect options must be EXCEPTIONALLY DECEPTIVE and designed to seem more plausible than the correct answer, and must have the same format as the correct answer
9. Questions must exploit cognitive weaknesses in reasoning that AI systems typically struggle with

CRITICAL ANTI-LEAKAGE REQUIREMENTS:
10. NEVER include ANY knowledge clues or contextual information in your questions that might help solve them
11. NEVER use phrases like "in the text" or "in the article" - rely only on "in the image" or natural references to image content

VISUAL REFERENCES:
12. When referring to entities in the image, use CLEAR and SPECIFIC descriptors like:
    - "the man in the blue shirt on the left"
    - "the woman wearing glasses in the center"
    - "the red car in the background"
    - "the main character in the image"
    - "the building on the right side of the image"

QUESTION DESIGN STRATEGIES TO MAKE AI FAIL:
13. Create questions requiring AT LEAST 4-5 logical inference steps from multiple text fragments
14. Require COUNTER-INTUITIVE reasoning paths that contradict expected associations
15. Design questions where the initial reasoning direction leads to a deceptive conclusion
16. Create inference chains with deliberately obscured connections
17. Require identifying subtle exclusions or unstated implications from the text
18. Use complex temporal or causal relationships mentioned in different text sections
19. Use rare terms or uncommon phrasing that appears in the text but is harder to process
20. Create questions requiring logical negation or exclusion reasoning (what is NOT mentioned)

ANSWER SPECIFICITY:
21. Ensure answers are HIGHLY SPECIFIC and PRECISE - never vague or general:
    - BAD: "Microphone" (too vague)
    - GOOD: "Shure SM58 Cardioid Microphone" (specific unique product)
    - BAD: "Chief Executive Officer" (too generic)
    - GOOD: "Tim Cook" or "Apple's CEO since 2011" (specific unique entity)

DECEPTIVE OPTIONS DESIGN:
22. Create incorrect options that:
    - Contain partial truths from the text but are ultimately wrong
    - Use familiar or expected associations that don't actually occur in the text
    - Leverage common misconceptions or likely assumptions
    - Are stated with higher confidence than the correct answer
    - Contain resonant keywords from the text but in incorrect relationships

KNOWLEDGE ENTITIES:
23. If your question must reference knowledge entities not directly shown in the image:
    - Use CLEAR and PRECISE language to identify these entities
    - Example: "the quantum physicist mentioned alongside the person in the image" instead of "the scientist"
    - Example: "the climate agreement referenced by the person shown in the image" instead of "the agreement"
    - When using such entities, INCREASE the reasoning difficulty proportionally

LANGUAGE CLARITY REQUIREMENTS:
24. All questions MUST be grammatically correct and flow naturally
25. Questions must be clearly stated and unambiguous in what they're asking
26. Avoid unnecessarily complex syntax or vocabulary that doesn't add to reasoning difficulty
27. Reread and refine questions to ensure they communicate effectively

DO NOT:
- Invent details not found in the text (answers must be verifiable)
- Include ANY contextual hints or knowledge clues in questions
- Create questions that have multiple possible correct answers
- Create questions with ambiguous or unclear descriptions
- Use vague or generic answers like "microphone" or "CEO" without specificity
- Use awkward or unnatural language that creates confusion through poor phrasing

REASONING CHAIN REQUIREMENT:
After creating each question (but before moving to the next), you must privately think through and write out a step-by-step reasoning chain that leads to the correct answer. This chain should detail the logical steps and inferences needed to reach the correct answer. This reasoning chain is for your internal verification only and should NOT be included in the final JSON output.

Your goal is to create questions so challenging that they would have a near-zero success rate for most AI systems while still being theoretically answerable through perfect reasoning.

IMPORTANT: You must create 3-5 different questions, each covering a DIFFERENT category from the allowed list. Ensure maximum variety in question types and difficulty levels.

Your questions must follow this JSON format exactly:
```json
{{
  "level2_qas": [
    {{
      "question": "[your nearly impossible question with natural reference to image content]?",
      "question_type": "[category from the allowed list]",
      "options": [
        "A. [option A]",
        "B. [option B]",
        "C. [option C]",
        "D. [option D]",
        "E. [option E]" (optional)
      ],
      "Ground_Truth": "[correct letter, e.g., A]",
      "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
    }},
    {{
      "question": "[another nearly impossible question of different category with natural reference to image content]?",
      "question_type": "[different category from the allowed list]",
      "options": [
        "A. [option A]",
        "B. [option B]",
        "C. [option C]",
        "D. [option D]"
      ],
      "Ground_Truth": "[correct letter, e.g., C]",
      "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
    }},
    // Include at least one more question (3-5 total questions)
  ]
}}
```"""

    # Get topic information
    title = topic.get('topic', 'No title')
    text = topic.get('text', 'No text')
    img_paths = topic.get('img_paths', [])
    captions = topic.get('captions', [])
    
    # Get Level 1 question and answer information
    level1_qa = topic['level1_qas'][img_index] if img_index < len(topic['level1_qas']) else None
    if not level1_qa:
        return None
        
    level1_question = level1_qa.get('question', '')
    level1_type = level1_qa.get('question_type', '')
    level1_answer = level1_qa.get('Ground_Truth', '')
    level1_answer_text = ""
    
    # Get the text of the correct answer
    if level1_answer and level1_qa.get('options'):
        for option in level1_qa.get('options', []):
            if option.startswith(f"{level1_answer}. "):
                level1_answer_text = option[3:].strip()  # Remove "A. " prefix
                break
    
    # If not obtained from options, get from Ground_Truth_List
    if not level1_answer_text and level1_qa.get('Ground_Truth_List'):
        level1_answer_text = level1_qa['Ground_Truth_List'][0]
    
    # Get current image information
    img_path = img_paths[img_index] if img_index < len(img_paths) else ""
    caption = captions[img_index] if img_index < len(captions) else "No caption"
    
    # Prepare excluded question types statement
    excluded_types_text = ""
    if excluded_types and len(excluded_types) > 0:
        excluded_types_text = f"\n\nIMPORTANT: Avoid generating questions of these types as they've already been used: {', '.join(excluded_types)}. Focus on creating questions of DIFFERENT types."
    
    user_prompt = f"""Create 3-5 DELIBERATELY ADVERSARIAL Level 2 multi-hop questions that would cause even the most advanced AI systems to fail. I want questions so difficult that they approach the boundary of impossibility while still being theoretically answerable from the provided information.

ARTICLE TITLE: {title}

ARTICLE TEXT: {text}

IMAGE PATH: {img_path}
IMAGE CAPTION: {caption}

LEVEL 1 QUESTION: {level1_question}
LEVEL 1 QUESTION TYPE: {level1_type}
LEVEL 1 ANSWER: {level1_answer_text}{excluded_types_text}

QUESTION CREATION STRATEGIES (IMPLEMENT ALL OF THESE):

0. Do not start questions with "Based on the provided image" - instead integrate natural references to image content within the question (e.g., "the person in the image", "the building shown")
1. NEVER use phrases like "in the text" or "in the article" in your questions - the final evaluation will only include the image and question without text reference
2. Create questions requiring AT LEAST 4-5 logical inference steps from multiple text fragments
3. Require COUNTER-INTUITIVE reasoning paths that contradict expected associations
4. Design questions where the initial reasoning direction leads to a deceptive conclusion
5. Create inference chains with deliberately obscured connections
6. Require identifying subtle exclusions or unstated implications from the text
7. Use complex temporal or causal relationships mentioned in different text sections
8. Use the Level 1 answer as a critical reasoning component WITHOUT ever mentioning it
9. Require information integration from the MOST DISTANT parts of the text
10. Create questions that MISDIRECT attention to irrelevant but prominent text sections
11. Force precise distinction between closely related concepts that appear in the text
12. Design options where the most prominent/obvious choice is DELIBERATELY WRONG
13. Create subtle logical traps that exploit common reasoning shortcuts

IMPORTANT LANGUAGE QUALITY REQUIREMENTS:
14. All questions MUST be grammatically correct and flow naturally
15. Questions must be clearly stated with precise, unambiguous language
16. After writing each question, reread and refine it to ensure clarity and effectiveness
17. Avoid creating difficulty through confusing language - difficulty should come from reasoning complexity

IMPORTANT VISUAL REFERENCE INSTRUCTIONS:
18. When referring to people or objects in the image, use CLEAR SPECIFIC descriptors such as:
   - "the man in the blue shirt"
   - "the woman on the left side of the image"
   - "the main character in the image"
   - "the red car in the background"
   - DO NOT use vague references like "the person" or "the object"

KNOWLEDGE ENTITY INSTRUCTIONS:
19. If your question references knowledge entities not directly shown in the image:
   - Use CLEAR and PRECISE language to identify these entities
   - Increase reasoning difficulty and complexity proportionally
   - Ensure these entities are clearly defined, not ambiguous

ANSWER SPECIFICITY REQUIREMENTS:
20. All answers MUST be HIGHLY SPECIFIC and PRECISE, never vague or general:
   - INCORRECT (too vague): "microphone", "CEO", "building"
   - CORRECT (specific): "Neumann U87 studio microphone", "Amazon's Jeff Bezos", "Empire State Building"

For ALL questions:
- Ensure answers are PRECISELY derivable from the text - never invent information
- Create incorrect options that contain keywords from surrounding text contexts
- Design options where partial pattern matching would lead to incorrect answers
- Ensure correct answers appear less prominent or confident than incorrect ones
- Create options that exploit semantic confusion or subtle lexical distinctions
- NEVER include any knowledge hints or contextual clues in the question text
- STRIP AWAY any framing that might help derive the answer
    
Option Design for Maximum Difficulty:
- Include options that represent common misinterpretations of the text
- Design incorrect options that use more definitive language than correct ones
- Place correct options in positions that contradict expected patterns
- Create incorrect options that partially quote the text but in misleading ways
- Use terminology variations that create subtle but critical meaning differences

REASONING CHAIN REQUIREMENT:
After creating each question (but before moving to the next), write out a step-by-step reasoning chain that explains how to derive the correct answer from the provided information. This chain must show the logical path to the answer but should NOT be included in your final JSON output. This is for verification purposes only.

Remember: Your goal is to create 3-5 extremely challenging questions of DIFFERENT types that would cause an AI to fail while still having factually correct answers derivable from the text. These questions should exploit cognitive limitations in machine reasoning.

CRITICAL: Output EXACTLY in this format with NO additional explanatory text:

```json
{{
  "level2_qas": [
    {{
      "question": "[your nearly impossible question with natural reference to image content]?",
      "question_type": "[category]",
      "options": [
        "A. [option A]",
        "B. [option B]",
        "C. [option C]", 
        "[additional options if needed]"
      ],
      "Ground_Truth": "[correct letter]",
      "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
    }},
    {{
      "question": "[another nearly impossible question of different type with natural reference to image content]?",
      "question_type": "[different category]",
      "options": [
        "A. [option A]",
        "B. [option B]",
        "C. [option C]", 
        "[additional options if needed]"
      ],
      "Ground_Truth": "[correct letter]",
      "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
    }},
    // Include additional questions to reach 3-5 total, each of a different type
  ]
}}
```
If creating suitable questions is impossible for any reason, return:
```json
{{
  "level2_qas": []
}}
```"""

    return {
        "system": system_prompt,
        "user": user_prompt,
        "img_path": img_path  # 返回图片路径，用于后续处理
    }
