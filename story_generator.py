import ollama
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORY_TEMPLATE = """Please generate a comic story with this exact structure based on: {premise}

1. TITLE: [Creative story title]
2. GENRE: [Story genre]
3. MAIN_CHARACTER:
   - name: [Character name]
   - appearance: [Physical description]
   - personality: [Key traits]
   - special: [Unique abilities/items]
4. SUPPORTING_CHARACTERS: [List other characters]
5. SETTING: [World description]
6. SCENES:
   - Scene 1:
     - description: [2-3 sentence scene]
     - visual_style: [Art style notes]
     - colors: [Color palette]
   - Scene 2:
     - description: [2-3 sentence scene]
     - visual_style: [Art style notes]
     - colors: [Color palette]
   [Continue for 5 scenes total]

Return ONLY the structured content, no additional commentary."""

def generate_story(premise: str) -> Dict:
    """Generate a complete story structure with validation"""
    try:
        logger.info(f"Generating story for premise: {premise}")
        response = ollama.chat(
            model="llama3.1:latest",
            messages=[{
                "role": "user",
                "content": STORY_TEMPLATE.format(premise=premise)
            }]
        )
        return _parse_story(response['message']['content'])
    except Exception as e:
        logger.error(f"Story generation failed: {str(e)}")
        return _default_story(premise)

def _parse_story(raw_text: str) -> Dict:
    """Parse the raw story text into a structured dictionary"""
    story = {
        "title": "Untitled Story",
        "genre": "Fantasy",
        "main_character": {
            "name": "Unknown",
            "appearance": "",
            "personality": "",
            "special": ""
        },
        "supporting_characters": [],
        "setting": "",
        "scenes": []
    }

    current_section = None
    current_scene = None
    
    for line in raw_text.split('\n'):
        line = line.strip()
        
        # Section headers
        if line.endswith(':'):
            current_section = line[:-1].lower().replace(" ", "_")
            if current_section.startswith("scene"):
                current_scene = int(current_section.split()[1]) - 1
                story["scenes"].append({
                    "description": "",
                    "visual_style": "",
                    "colors": ""
                })
            continue
        
        # Content parsing
        if current_section:
            if current_section == "main_character":
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip().lower()
                    if key in story["main_character"]:
                        story["main_character"][key] = val.strip()
            elif current_section == "scenes" and current_scene is not None:
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip().lower().replace(" ", "_")
                    if key in story["scenes"][current_scene]:
                        story["scenes"][current_scene][key] = val.strip()
            else:
                if current_section in story:
                    story[current_section] += line + "\n"
                elif current_section.replace(" ", "_") in story:
                    story[current_section.replace(" ", "_")] += line + "\n"
    
    return story

def _default_story(premise: str) -> Dict:
    """Fallback story structure if generation fails"""
    return {
        "title": premise[:30] + "...",
        "genre": "Adventure",
        "main_character": {
            "name": "Hero",
            "appearance": "A brave protagonist",
            "personality": "Courageous and kind",
            "special": "Special item or ability"
        },
        "supporting_characters": ["Sidekick"],
        "setting": "A mysterious world",
        "scenes": [
            {
                "description": f"{premise}. The story begins...",
                "visual_style": "Cartoon style",
                "colors": "Vibrant colors"
            }
        ] * 3  # Create 3 default scenes
    }