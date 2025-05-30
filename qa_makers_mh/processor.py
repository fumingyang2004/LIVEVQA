import logging
import traceback

from qa_makers_mh.prompt_generator import create_multihop_prompt
from qa_makers_mh.api_client import generate_multihop_questions

logger = logging.getLogger(__name__)

def process_topic(client, topic, results, index):
    """
    Processes a single topic entry to generate multi-hop questions.
    
    Args:
        client: The OpenAI client.
        topic: The topic data.
        results: The list of results (will be modified).
        index: The index of the topic in the results list.
        
    Returns:
        bool: True if processed successfully, False otherwise.
    """
    # Skip discarded topics
    if topic.get('discarded', False):
        logger.info(f"Skipping discarded topic ID: {topic.get('id')}")
        return True
    
    # Check for level1_qas
    if not topic.get('level1_qas'):
        logger.warning(f"Topic ID {topic.get('id')} has no level1_qas, skipping.")
        # Create result object without any multi-hop questions
        topic_copy = topic.copy()
        results[index] = topic_copy
        return True
    
    # Create a copy of the original topic for results
    topic_copy = topic.copy()
    
    # Process questions for all images
    try:
        # Get the number of level1_qas
        level1_qas_count = len(topic.get('level1_qas', []))
        
        # Process each level1 question to create corresponding level2 questions
        for img_index in range(level1_qas_count):
            # Create prompt
            prompt_data = create_multihop_prompt(topic, img_index)
            if not prompt_data:
                logger.warning(f"Could not create prompt for topic ID {topic.get('id')} image {img_index}")
                continue
            
            # Generate multi-hop questions
            multihop_questions = generate_multihop_questions(client, prompt_data)
            
            # Get level2 questions
            level2_qas = multihop_questions.get('level2_qas', [])
            
            # Get corresponding img_path and img_url
            level1_qa = topic['level1_qas'][img_index]
            img_path = level1_qa.get('img_path', '')
            img_url = level1_qa.get('img_url', '')
            
            # Check if enough questions were generated
            if len(level2_qas) < 3:
                # If too few questions, try generating again
                logger.warning(f"Insufficient questions generated for topic ID {topic.get('id')} image {img_index} ({len(level2_qas)} < 3), attempting to re-generate.")
                
                # Try generating up to 2 extra times
                attempts = 0
                max_attempts = 2
                
                while len(level2_qas) < 3 and attempts < max_attempts:
                    # Generate additional questions, requesting different types
                    excluded_types = [q.get('question_type', '').lower() for q in level2_qas]
                    additional_prompt_data = create_multihop_prompt(topic, img_index, excluded_types=excluded_types)
                    additional_questions = generate_multihop_questions(client, additional_prompt_data)
                    additional_level2_qas = additional_questions.get('level2_qas', [])
                    
                    # Filter for non-duplicate question types
                    used_types = set(q.get('question_type', '').lower() for q in level2_qas)
                    
                    for qa in additional_level2_qas:
                        qa_type = qa.get('question_type', '').lower()
                        # If this question type is new or has few instances, and total is less than 5, add it
                        if qa_type not in used_types and len(level2_qas) < 5:
                            level2_qas.append(qa)
                            used_types.add(qa_type)
                    
                    attempts += 1
                
                if len(level2_qas) < 3:
                    logger.warning(f"Topic ID {topic.get('id')} image {img_index} ultimately generated only {len(level2_qas)} questions, fewer than the target of 3.")
            elif len(level2_qas) > 5:
                # If too many questions, truncate to the first 5
                logger.info(f"Too many questions generated for topic ID {topic.get('id')} image {img_index} ({len(level2_qas)} > 5), truncating to the first 5.")
                
                # Ensure the truncated questions are as diverse as possible by type
                question_types = {}
                filtered_qas = []
                
                for qa in level2_qas:
                    qa_type = qa.get('question_type', '').lower()
                    
                    # If this type of question isn't present yet or there are few of this type, add it
                    if qa_type not in question_types:
                        question_types[qa_type] = 1
                        filtered_qas.append(qa)
                    elif question_types[qa_type] < 2 and len(filtered_qas) < 5:
                        question_types[qa_type] += 1
                        filtered_qas.append(qa)
                    
                    if len(filtered_qas) >= 5:
                        break
                
                level2_qas = filtered_qas
            
            # Add img_path and img_url to each level2 question
            for qa in level2_qas:
                qa['img_path'] = img_path
                qa['img_url'] = img_url
            
            # Add to the results object using the correct key (level2_qas_img1, level2_qas_img2, etc.)
            img_key = f'level2_qas_img{img_index+1}'
            topic_copy[img_key] = level2_qas
            
            logger.info(f"Generated {len(level2_qas)} level2 questions for topic ID {topic.get('id')} image {img_index+1}.")
        
        # Update results
        results[index] = topic_copy
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing topic ID {topic.get('id')}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def process_topic_thread(args):
    """
    Thread function for processing a single topic.
    
    Args:
        args: A tuple containing processing arguments.
        
    Returns:
        bool: True if processed successfully, False otherwise.
    """
    client, topic, results, index = args
    try:
        return process_topic(client, topic, results, index)
    except Exception as e:
        logger.error(f"Error processing topic in thread: {str(e)}")
        logger.error(traceback.format_exc())
        return False