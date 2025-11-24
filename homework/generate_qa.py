import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    """
    with open(info_path) as f:
        info = json.load(f)

    if "detections" not in info or view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]
    
    # Get kart names list
    kart_names = info.get("karts", [])

    cart_list = []
    for det in frame_detections:
        try:
            class_id, track_id, x1, y1, x2, y2 = det
        except Exception:
            continue

        class_id = int(class_id)
        track_id = int(track_id)

        # only consider karts (class 1)
        if class_id != 1:
            continue

        # scale coordinates
        cx = (x1 + x2) / 2.0 * (img_width / ORIGINAL_WIDTH)
        cy = (y1 + y2) / 2.0 * (img_height / ORIGINAL_HEIGHT)

        w = (x2 - x1) * (img_width / ORIGINAL_WIDTH)
        h = (y2 - y1) * (img_height / ORIGINAL_HEIGHT)

        # filter tiny boxes or out-of-bounds
        if w < min_box_size or h < min_box_size:
            continue

        if cx < 0 or cy < 0 or cx > img_width or cy > img_height:
            continue

        # Get kart name from list using track_id as index
        kart_name = kart_names[track_id] if track_id < len(kart_names) else f"kart_{track_id}"

        cart_list.append({
            "instance_id": track_id, 
            "kart_name": kart_name, 
            "center": (cx, cy), 
            "bbox": (x1, y1, x2, y2)
        })

    # determine center (ego) kart
    if not cart_list:
        return []

    img_cx = img_width / 2.0
    img_cy = img_height / 2.0

    dists = [((c["center"][0] - img_cx) ** 2 + (c["center"][1] - img_cy) ** 2, i) for i, c in enumerate(cart_list)]
    dists.sort()
    center_idx = dists[0][1]

    for i, c in enumerate(cart_list):
        c["is_center_kart"] = i == center_idx

    return cart_list


# Add this function before the main() function:

def generate(data_dir: str, output_file: str = None, num_views: int = 10):
    """
    Generate QA pairs for all info files in a directory.
    
    Args:
        data_dir: Directory containing *_info.json files
        output_file: Output JSON file path (default: data_dir/balanced_qa_pairs.json)
        num_views: Number of views per frame (default: 10)
    """
    from tqdm import tqdm
    
    data_dir = Path(data_dir)
    
    if output_file is None:
        output_file = data_dir / "balanced_qa_pairs.json"
    else:
        output_file = Path(output_file)
    
    # Find all info files
    info_files = sorted(data_dir.glob("*_info.json"))
    
    if len(info_files) == 0:
        print(f"❌ No *_info.json files found in {data_dir}")
        return
    
    print(f"📁 Found {len(info_files)} info files in {data_dir}")
    
    all_qa_pairs = []
    
    # Process each info file and each view
    for info_file in tqdm(info_files, desc="Generating QA pairs"):
        try:
            # First check how many views this file actually has
            with open(info_file) as f:
                info_data = json.load(f)
            actual_num_views = len(info_data.get("detections", []))
            
            for view_idx in range(min(num_views, actual_num_views)):
                # Check if image exists for this view
                base_name = info_file.stem.replace("_info", "")
                image_file = info_file.parent / f"{base_name}_{view_idx:02d}_im.jpg"
                
                if not image_file.exists():
                    continue
                
                # Generate QA pairs for this view
                qa_pairs = generate_qa_pairs(str(info_file), view_idx)
                all_qa_pairs.extend(qa_pairs)
                
        except Exception as e:
            print(f"\n⚠️  Error processing {info_file.name}: {e}")
            continue
    
    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\n✅ Generated {len(all_qa_pairs)} QA pairs")
    print(f"📝 Saved to: {output_file}")
    print(f"\n📊 Statistics:")
    print(f"   - Info files processed: {len(info_files)}")
    print(f"   - Total QA pairs: {len(all_qa_pairs)}")
    print(f"   - Avg QA pairs per file: {len(all_qa_pairs) / len(info_files):.1f}")
    
    return all_qa_pairs

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)

    # common keys
    for key in ("track_name", "track", "map_name", "map"):
        if key in info:
            return str(info[key])

    # fallback to unknown
    return "unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    # prepare image file path relative to info file
    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_name = f"{base_name}_{view_index:02d}_im.jpg"

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    qa_pairs = []

    # 1. Ego car question
    if karts:
        ego = next((k for k in karts if k.get("is_center_kart")), karts[0])
        qa_pairs.append({"question": "What kart is the ego car?", "answer": ego["kart_name"], "image_file": str(Path(info_path_obj.parent.name) / image_name)})
    else:
        qa_pairs.append({"question": "What kart is the ego car?", "answer": "none", "image_file": str(Path(info_path_obj.parent.name) / image_name)})

    # 2. Total karts question
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts)), "image_file": str(Path(info_path_obj.parent.name) / image_name)})

    # 3. Track information
    qa_pairs.append({"question": "What track is this?", "answer": track_name, "image_file": str(Path(info_path_obj.parent.name) / image_name)})

    # 4. Relative position questions
    if karts and len(karts) > 1:
        ego = next((k for k in karts if k.get("is_center_kart")), karts[0])
        ex, ey = ego["center"]

        left_count = 0
        right_count = 0
        front_count = 0
        behind_count = 0

        for k in karts:
            if k is ego:
                continue
            x, y = k["center"]
            # left/right by x coordinate
            if x < ex:
                lr = "left"
                left_count += 1
            else:
                lr = "right"
                right_count += 1

            # front/behind by y coordinate (smaller y -> front)
            if y < ey:
                fb = "front"
                front_count += 1
            else:
                fb = "behind"
                behind_count += 1

            qa_pairs.append({"question": f"Is {k['kart_name']} to the left or right of the ego car?", "answer": lr, "image_file": str(Path(info_path_obj.parent.name) / image_name)})
            qa_pairs.append({"question": f"Is {k['kart_name']} in front of or behind the ego car?", "answer": fb, "image_file": str(Path(info_path_obj.parent.name) / image_name)})
            qa_pairs.append({"question": f"Where is {k['kart_name']} relative to the ego car?", "answer": f"{fb} and to the {lr}", "image_file": str(Path(info_path_obj.parent.name) / image_name)})

        # 5. Counting questions
        qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count), "image_file": str(Path(info_path_obj.parent.name) / image_name)})
        qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count), "image_file": str(Path(info_path_obj.parent.name) / image_name)})
        qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count), "image_file": str(Path(info_path_obj.parent.name) / image_name)})
        qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(behind_count), "image_file": str(Path(info_path_obj.parent.name) / image_name)})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate,
    })


if __name__ == "__main__":
    main()
