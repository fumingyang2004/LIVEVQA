import os
import json
import requests
import xml.etree.ElementTree as ET
import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image
import glob
import re
import uuid

def get_arxiv_abstract(paper_id):
    """Fetch paper abstract via arXiv API"""
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse XML response
            root = ET.fromstring(response.content)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                if summary is not None:
                    return summary.text.strip()
    except Exception as e:
        st.error(f"Error fetching abstract: {e}")

    return "Abstract not available"

def find_json_files(base_dir, pattern="*_selected_images.json"):
    """Find all JSON files matching the pattern and check for corresponding associations.json"""
    base_path = Path(base_dir)
    if not base_path.exists():
        st.warning(f"Warning: Directory {base_dir} does not exist")
        return []

    # Use Path.glob to recursively find all matching files
    matching_files = list(base_path.glob(f"**/{pattern}"))
    valid_files = []

    for selected_file in matching_files:
        # Check if associations.json exists in the same directory
        association_path = selected_file.parent / "associations.json"
        if association_path.exists():
            valid_files.append(str(selected_file))

    # Modified: Sort files by dictionary order (alphabetical)
    valid_files.sort()

    return valid_files

def correct_image_path(json_path, img_filename):
    """Correct image path"""
    # Extract year and paper ID from JSON path
    json_path = Path(json_path)

    # Extract year and paper ID
    match = re.search(r'(\d{4}-\d{2}-\d{2})/([\d\.]+)', str(json_path))
    if match:
        year_month = match.group(1)
        paper_id = match.group(2)

        # Construct image path
        img_path = Path(f"/media/sata3/cdp/livevqa-arxiv/data/raw/images/{year_month}/{paper_id}/{img_filename}")
        if img_path.exists():
            return img_path

    # Fallback strategy 1: Look for the image in the parent directory
    parent_dir = json_path.parent
    img_path = parent_dir / img_filename
    if img_path.exists():
        return img_path

    # Fallback strategy 2: Look in the 'images' directory at the same level
    img_path = parent_dir.parent / "images" / parent_dir.name / img_filename
    if img_path.exists():
        return img_path

    return None

def load_paper_data(selected_json_path, association_json_path=None):
    """Load paper data from both selected_images.json and associations.json"""
    data = {}

    # Load selected_images.json
    try:
        with open(selected_json_path, 'r', encoding='utf-8') as f:
            selected_data = json.load(f)
            data.update(selected_data)
    except Exception as e:
        st.error(f"Failed to load selected_images.json file: {e}")
        return None

    # If association_json_path is not provided, try to find it automatically
    if association_json_path is None:
        association_json_path = Path(selected_json_path).parent / "associations.json"

    # Load associations.json
    try:
        if Path(association_json_path).exists():
            with open(association_json_path, 'r', encoding='utf-8') as f:
                association_data = json.load(f)
                # If association contains abstract, use it
                if 'abstract' in association_data:
                    data['abstract_from_association'] = association_data['abstract']
                # Merge other data
                for key, value in association_data.items():
                    if key not in data or key == 'authors' or key == 'paragraphs':
                        data[key] = value
    except Exception as e:
        st.warning(f"Failed to load associations.json file: {e}")

    return data

def save_feedback(json_file, image_index, is_selected, is_reasonable):
    """Save user feedback on image selection reasonableness"""
    try:
        feedback_dir = Path("/media/sata3/cdp/livevqa-arxiv/feedback")
        feedback_dir.mkdir(exist_ok=True)

        paper_id = Path(json_file).stem.split("_")[0]
        feedback_file = feedback_dir / f"{paper_id}_feedback.json"

        # Load existing feedback or create new feedback
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = {"paper_id": paper_id, "feedbacks": []}

        # Add new feedback
        feedback_item = {
            "image_index": image_index,
            "is_selected": is_selected,
            "is_reasonable": is_reasonable,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Check if feedback for the same image already exists, update if so
        updated = False
        for i, item in enumerate(feedback_data["feedbacks"]):
            if item["image_index"] == image_index and item["is_selected"] == is_selected:
                feedback_data["feedbacks"][i] = feedback_item
                updated = True
                break

        if not updated:
            feedback_data["feedbacks"].append(feedback_item)

        # Save feedback
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")
        return False

def display_figure(json_path, figure, i, is_selected, session_id):
    """Display a single image and provide reasonableness marking function"""
    # Generate a unique ID for this image
    figure_id = f"{session_id}_{figure.get('index', i)}"

    # Prepare image path
    img_path = figure.get('img_local_path')
    if not img_path:
        # If image_local_path is not available, try using image_url and correct_image_path function
        img_filename = Path(figure.get('image_url', '')).name
        img_path = correct_image_path(json_path, img_filename)
    else:
        # Convert string to Path object
        img_path = Path(img_path)

    # Determine caption text
    selection_status = "Selected" if is_selected else "Not Selected"
    caption = f"Image {figure.get('index', i)+1}: {Path(img_path).name} (Rank: {figure.get('rank')}) - {selection_status}"

    with st.expander(caption, expanded=is_selected):
        col1, col2 = st.columns([3, 2])

        with col1:
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=f"Image Index: {figure.get('index', i)}", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                    st.write(f"Image Path: {img_path}")
            else:
                st.error(f"Image file not found: {img_path}")
                st.write(f"Attempted path: {img_path}")

        with col2:
            st.markdown("**Image Caption:**")
            st.write(figure.get('caption', 'No caption'))

            if 'reason' in figure:
                st.markdown("**Selection/Scoring Reason:**")
                st.write(figure.get('reason'))

            # Add reasonableness evaluation option
            st.markdown("**Do you think it is reasonable that this image was {}?**".format("selected" if is_selected else "not selected"))

            # Get previous feedback status
            feedback_key = f"feedback_{figure_id}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None

            # Create callback function for automatic feedback saving
            def on_feedback_change():
                feedback_value = st.session_state[feedback_key]
                if feedback_value is not None:
                    is_reasonable = (feedback_value == "Reasonable")
                    save_feedback(json_path, figure.get('index', i), is_selected, is_reasonable)

            # Use radio buttons instead of buttons, save automatically after selection
            # Removed feedback_options variable as it's not used directly in st.radio options
            feedback_value = st.radio(
                "Select evaluation:",
                options=["Reasonable", "Unreasonable"],
                key=feedback_key,
                on_change=on_feedback_change,
                horizontal=True,
                index=None if st.session_state[feedback_key] is None else (0 if st.session_state[feedback_key] == "Reasonable" else 1)
            )

            # Display feedback status
            if st.session_state[feedback_key] is not None:
                st.success(f"Your feedback has been recorded: {st.session_state[feedback_key]}")

def get_feedback_stats(json_file):
    """Get user feedback statistics: reasonable ratio, total marked, etc."""
    try:
        feedback_dir = Path("/media/sata3/cdp/livevqa-arxiv/feedback")
        paper_id = Path(json_file).stem.split("_")[0]
        feedback_file = feedback_dir / f"{paper_id}_feedback.json"

        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)

            feedbacks = feedback_data.get("feedbacks", [])
            total_feedback = len(feedbacks)

            if total_feedback == 0:
                return {"total": 0, "reasonable": 0, "unreasonable": 0, "reasonable_ratio": 0}

            reasonable_count = sum(1 for item in feedbacks if item.get("is_reasonable", False))
            unreasonable_count = total_feedback - reasonable_count
            reasonable_ratio = reasonable_count / total_feedback if total_feedback > 0 else 0

            return {
                "total": total_feedback,
                "reasonable": reasonable_count,
                "unreasonable": unreasonable_count,
                "reasonable_ratio": reasonable_ratio
            }

        return {"total": 0, "reasonable": 0, "unreasonable": 0, "reasonable_ratio": 0}
    except Exception as e:
        st.error(f"Failed to get feedback statistics: {e}")
        return {"total": 0, "reasonable": 0, "unreasonable": 0, "reasonable_ratio": 0}

def get_all_feedback_stats():
    """Get feedback statistics for all papers"""
    try:
        feedback_dir = Path("/media/sata3/cdp/livevqa-arxiv/feedback")
        if not feedback_dir.exists():
            return {"total": 0, "reasonable": 0, "unreasonable": 0, "reasonable_ratio": 0}

        feedback_files = list(feedback_dir.glob("*_feedback.json"))

        total_feedback = 0
        total_reasonable = 0

        for feedback_file in feedback_files:
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)

                feedbacks = feedback_data.get("feedbacks", [])
                total_feedback += len(feedbacks)
                total_reasonable += sum(1 for item in feedbacks if item.get("is_reasonable", False))
            except Exception as e:
                st.warning(f"Failed to read feedback file {feedback_file}: {e}")

        unreasonable_count = total_feedback - total_reasonable
        reasonable_ratio = total_reasonable / total_feedback if total_feedback > 0 else 0

        return {
            "total": total_feedback,
            "reasonable": total_reasonable,
            "unreasonable": unreasonable_count,
            "reasonable_ratio": reasonable_ratio
        }
    except Exception as e:
        st.error(f"Failed to get all feedback statistics: {e}")
        return {"total": 0, "reasonable": 0, "unreasonable": 0, "reasonable_ratio": 0}

def display_paper_info(json_file):
    """Display paper information, selected images, and unselected images"""
    try:
        json_path = Path(json_file)
        if not json_path.exists():
            st.error(f"Error: JSON file not found: {json_file}")
            return

        # Get path to the associated associations.json
        association_path = json_path.parent / "associations.json"
        if not association_path.exists():
            st.error(f"Error: associations.json file not found: {association_path}")
            return

        # Load paper data (read from both files)
        data = load_paper_data(json_path, association_path)
        if not data:
            st.error("Failed to load paper data")
            return

        paper_id = data.get('paper_id')
        title = data.get('title')
        selected_figures = data.get('selected_figures', [])
        all_figures = data.get('all_figures', [])
        recommended_count = data.get('recommended_count', len(selected_figures))
        selection_reason = data.get('selection_reason', 'Reason not provided')

        if not paper_id or not title:
            st.error("JSON file is missing paper ID or title")
            return

        # Create a unique session ID for this paper, avoiding button ID conflicts for different images
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        session_id = st.session_state.session_id

        # Get abstract and display basic information
        st.header(title)
        with st.expander("Paper Details", expanded=True):
            st.write(f"**Paper ID:** {paper_id}")

            # Display author information
            if 'authors' in data:
                authors = data.get('authors', [])
                st.write(f"**Authors:** {', '.join(authors)}")

            # Add arXiv link
            arxiv_link = f"https://arxiv.org/abs/{paper_id}"
            st.markdown(f"**arXiv Link:** [View on arXiv]({arxiv_link})")

            # Prioritize abstract from association
            abstract = data.get('abstract_from_association')
            if not abstract:
                abstract = data.get('abstract')
            if not abstract:
                abstract = get_arxiv_abstract(paper_id)

            st.markdown("### Abstract")
            st.write(abstract)

        # Get feedback statistics
        feedback_stats = get_feedback_stats(json_file)

        # Display feedback statistics
        st.markdown("### Image Selection Reasonableness Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Marked", feedback_stats["total"])
        with col2:
            st.metric("Marked Reasonable", feedback_stats["reasonable"])
        with col3:
            st.metric("Marked Unreasonable", feedback_stats["unreasonable"])
        with col4:
            reasonable_percent = f"{feedback_stats['reasonable_ratio']:.1%}"
            st.metric("Reasonable Ratio", reasonable_percent)

        # Display image selection count and reason
        st.markdown("### Image Selection Quantity Explanation")
        st.write(f"**Recommended Selection Count:** {recommended_count} images")
        st.write(f"**Selection Reason:** {selection_reason}")

        # Build list of unselected images (exclude selected figures from all_figures)
        selected_indices = [fig.get('index') for fig in selected_figures]
        unselected_figures = [fig for fig in all_figures if fig.get('index') not in selected_indices]

        # Display selected images
        if selected_figures:
            st.markdown(f"### Selected Images ({len(selected_figures)})")
            for i, figure in enumerate(selected_figures):
                display_figure(json_path, figure, i, True, session_id)
        else:
            st.info("No selected images")

        # Display unselected images
        if unselected_figures:
            st.markdown(f"### Unselected Images ({len(unselected_figures)})")
            for i, figure in enumerate(unselected_figures):
                display_figure(json_path, figure, i, False, session_id)
        else:
            st.info("No unselected images")

    except Exception as e:
        st.error(f"Error processing JSON file: {e}")

def main():
    st.set_page_config(page_title="Paper Image Viewer", layout="wide")
    st.title("Paper Abstract and Image Viewer")

    # Display global feedback statistics
    all_stats = get_all_feedback_stats()
    st.sidebar.markdown("### Global Feedback Statistics")
    st.sidebar.metric("Total Marked", all_stats["total"])
    st.sidebar.metric("Reasonable Ratio", f"{all_stats['reasonable_ratio']:.1%}")

    # Sidebar settings
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    data_dir = st.sidebar.text_input("Data Directory Path", "/media/sata3/cdp/livevqa-arxiv/data/processed/json/2025-05-01")
    search_btn = st.sidebar.button("Search Papers")

    # Store state
    if 'json_files' not in st.session_state:
        st.session_state.json_files = []

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Search papers
    if search_btn:
        st.session_state.json_files = find_json_files(data_dir)
        st.session_state.current_index = 0

        if not st.session_state.json_files:
            st.error(f"No matching JSON files found in {data_dir} or missing required associations.json files")
        else:
            st.success(f"Found {len(st.session_state.json_files)} complete paper data entries")

    # Display navigation buttons
    if st.session_state.json_files:
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

        with col1:
            if st.button("⏮️ First Paper"):
                st.session_state.current_index = 0

        with col2:
            if st.button("⏪ Previous Paper") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1

        with col3:
            st.write(f"Current: {st.session_state.current_index + 1} / {len(st.session_state.json_files)}")

        with col4:
            if st.button("Next Paper ⏩") and st.session_state.current_index < len(st.session_state.json_files) - 1:
                st.session_state.current_index += 1

        # Display current paper information
        if 0 <= st.session_state.current_index < len(st.session_state.json_files):
            json_file = st.session_state.json_files[st.session_state.current_index]
            display_paper_info(json_file)

if __name__ == "__main__":
    main()