from fpdf import FPDF
from datetime import datetime
from typing import Dict, List

def create_comic_pdf(story: Dict, image_paths: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    
    # Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, story["title"], 0, 1, "C")
    pdf.set_font("Helvetica", "I", 16)
    pdf.cell(0, 10, f"A {story['genre']} Adventure", 0, 1, "C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, story["summary"])
    
    # Character Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "Main Character", 0, 1)
    pdf.set_font("Helvetica", "", 12)
    char = story["main_character"]
    pdf.multi_cell(0, 8, 
        f"Name: {char['name']}\n"
        f"Appearance: {char['appearance']}\n"
        f"Personality: {char.get('personality', '')}\n"
        f"Special: {char['special']}"
    )
    
    # Scenes
    for i, (scene, img_path) in enumerate(zip(story["scenes"], image_paths)):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Scene {i+1}", 0, 1)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, scene["description"])
        pdf.image(img_path, x=20, w=170)
    
    return pdf.output(dest="S").encode("latin1")