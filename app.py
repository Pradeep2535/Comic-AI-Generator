import streamlit as st
from image_generator import ImageGenerator
from story_generator import generate_story, _default_story  # Explicitly import _default_story
from pdf_export import create_comic_pdf
from PIL import Image
import time
import gc
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Setup -------------------
def configure_memory():
    """Configure PyTorch memory settings"""
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

# ------------------- UI Components -------------------
def init_sidebar():
    """Initialize sidebar controls"""
    with st.sidebar:
        st.title("⚙️ Comic Generator")
        
        st.subheader("Story Setup")
        premise = st.text_area(
            "Your Story Idea:",
            "A robot detective solves mysteries in a futuristic city",
            height=100
        )
        
        st.subheader("Generation Settings")
        low_memory = st.checkbox("Low Memory Mode", True)
        strength = st.slider("Image Strength", 0.3, 0.8, 0.6)
        steps = st.slider("Generation Steps", 15, 40, 25)
        batch_size = st.slider("Scenes per Batch", 1, 3, 1) if low_memory else 3
        
        if st.button("Generate Comic", type="primary", use_container_width=True):
            return {
                'premise': premise,
                'low_memory': low_memory,
                'strength': strength,
                'steps': steps,
                'batch_size': batch_size
            }
    return None

def display_character_sheet(character):
    """Display character information"""
    with st.expander("🧙 Character Sheet", expanded=True):
        cols = st.columns([1, 2])
        with cols[0]:
            if 'image' in character:
                st.image(character['image'], use_column_width=True)
        
        with cols[1]:
            st.subheader(character.get('name', 'Unknown'))
            st.markdown(f"**Appearance:** {character.get('appearance', 'Not specified')}")
            st.markdown(f"**Personality:** {character.get('personality', 'Not specified')}")
            st.markdown(f"**Special:** {character.get('special', 'None')}")

# ------------------- Main App -------------------
def main():
    # Initial configuration
    configure_memory()
    
    st.set_page_config(
        page_title="AI Comic Factory",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.title("🎨 AI Comic Factory")
    st.caption("Generate complete comic stories with consistent characters")
    
    # Initialize session state
    if 'generator' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.generator = ImageGenerator()
            try:
                st.session_state.generator.initialize_pipelines()
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                st.stop()
    
    # Get user inputs
    params = init_sidebar()
    
    if params:
        with st.status("🚀 Generating your comic...", expanded=True) as status:
            try:
                # Step 1: Generate story
                status.write("📖 Generating story outline...")
                try:
                    story = generate_story(params['premise'])
                    if not story.get('main_character') or not story.get('scenes'):
                        raise ValueError("Incomplete story generated")
                except Exception as e:
                    logger.warning(f"Using fallback story: {str(e)}")
                    story = _default_story(params['premise'])
                    st.warning("Used fallback story content due to generation issues")
                
                # Step 2: Create main character
                status.write("🎨 Creating main character...")
                char_desc = (
                    f"{story['main_character'].get('name', 'Hero')}, "
                    f"{story['main_character'].get('appearance', 'A mysterious character')}, "
                    f"{story['main_character'].get('special', 'Has special abilities')}"
                )
                
                character_img = st.session_state.generator.generate_character(
                    character_description=char_desc,
                    steps=params['steps'],
                    guidance=7.5
                )
                
                # Store character data
                character = {
                    **story['main_character'],
                    'image': character_img
                }
                
                # Step 3: Generate scenes in batches
                status.write("🖼️ Generating comic scenes...")
                scene_images = []
                scenes = story['scenes'][:5]  # Only use first 5 scenes
                
                for i in range(0, len(scenes), params['batch_size']):
                    batch = scenes[i:i+params['batch_size']]
                    for scene in batch:
                        img = st.session_state.generator.generate_scene(
                            base_image=character_img,
                            scene_description=scene['description'],
                            strength=params['strength'],
                            steps=params['steps'],
                            guidance=7.0
                        )
                        scene_images.append(img)
                        gc.collect()  # Manual garbage collection
                
                # Step 4: Create PDF
                status.write("📄 Creating printable PDF...")
                pdf_bytes = create_comic_pdf(story, scene_images)
                
                # Store results
                st.session_state.result = {
                    'story': story,
                    'character': character,
                    'scenes': list(zip(scenes, scene_images)),
                    'pdf': pdf_bytes
                }
                
                status.update(label="✅ Generation complete!", state="complete")
                
            except Exception as e:
                status.error(f"Generation failed: {str(e)}")
                st.session_state.generator.clear_cache()
                return
    
        # Display results
        if 'result' in st.session_state:
            result = st.session_state.result
            
            # Show story header
            st.subheader(result['story']['title'])
            st.caption(f"Genre: {result['story']['genre']}")
            
            # Display character sheet
            display_character_sheet(result['character'])
            
            # Show comic scenes
            st.divider()
            st.subheader("📜 Your Comic Story")
            
            for i, (scene, img) in enumerate(result['scenes']):
                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(img, use_column_width=True)
                    with cols[1]:
                        st.markdown(f"#### Scene {i+1}")
                        st.write(scene['description'])
                        st.caption(f"**Style:** {scene.get('visual_style', '')} | **Colors:** {scene.get('colors', '')}")
            
            # Download section
            st.divider()
            st.subheader("📥 Export Options")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download PDF Comic",
                    data=result['pdf'],
                    file_name=f"{result['story']['title']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                with st.expander("Advanced Downloads"):
                    # Character image download
                    img_bytes = io.BytesIO()
                    result['character']['image'].save(img_bytes, format='PNG')
                    st.download_button(
                        label="Download Character",
                        data=img_bytes.getvalue(),
                        file_name="character.png",
                        mime="image/png"
                    )
                    
                    # Scene images download
                    for i, (_, img) in enumerate(result['scenes']):
                        scene_bytes = io.BytesIO()
                        img.save(scene_bytes, format='PNG')
                        st.download_button(
                            label=f"Download Scene {i+1}",
                            data=scene_bytes.getvalue(),
                            file_name=f"scene_{i+1}.png",
                            mime="image/png"
                        )

if __name__ == "__main__":
    main()