from pathlib import Path

import fire
from matplotlib import pyplot as plt

from generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    # Reuse helpers from generate_qa
    from generate_qa import extract_kart_objects, extract_track_info

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    captions = []
    # Ego car
    if karts:
        ego = next((k for k in karts if k.get("is_center_kart")), karts[0])
        captions.append(f"{ego['kart_name']} is the ego car.")
    else:
        captions.append("There is no ego car visible.")

    # Counting
    captions.append(f"There are {len(karts)} karts in the scene.")

    # Track
    captions.append(f"The track is {track_name}.")

    # Relative positions for a few karts
    if karts and len(karts) > 1:
        ego = next((k for k in karts if k.get("is_center_kart")), karts[0])
        ex, ey = ego["center"]
        added = 0
        for k in karts:
            if k is ego:
                continue
            x, y = k["center"]
            lr = "left" if x < ex else "right"
            fb = "in front" if y < ey else "back"
            captions.append(f"{k['kart_name']} is {fb} and to the {lr} of the ego car.")
            added += 1
            if added >= 3:
                break

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

def generate(data_dir: str, output_file: str = None, num_views: int = 10):
    """
    Generate captions for all info files in a directory.
    
    Args:
        data_dir: Directory containing *_info.json files
        output_file: Output JSON file path (default: data_dir/balanced_captions.json)
        num_views: Number of views per frame (default: 10)
    """
    from tqdm import tqdm
    import json
    
    data_dir = Path(data_dir)
    
    if output_file is None:
        output_file = data_dir / "balanced_captions.json"
    else:
        output_file = Path(output_file)
    
    # Find all info files
    info_files = sorted(data_dir.glob("*_info.json"))
    
    if len(info_files) == 0:
        print(f"❌ No *_info.json files found in {data_dir}")
        return
    
    print(f"📁 Found {len(info_files)} info files in {data_dir}")
    
    all_captions = []
    
    # Process each info file and each view
    for info_file in tqdm(info_files, desc="Generating captions"):
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
                
                # Generate captions for this view
                captions = generate_caption(str(info_file), view_idx)
                
                # Combine into a single caption
                combined_caption = " ".join(captions)
                
                # Store with image path
                info_path_obj = Path(info_file)
                image_name = f"{base_name}_{view_idx:02d}_im.jpg"
                
                all_captions.append({
                    "image_file": str(Path(info_path_obj.parent.name) / image_name),
                    "caption": combined_caption
                })
                
        except Exception as e:
            print(f"\n⚠️  Error processing {info_file.name}: {e}")
            continue
    
    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"\n✅ Generated {len(all_captions)} captions")
    print(f"📝 Saved to: {output_file}")
    print(f"\n📊 Statistics:")
    print(f"   - Info files processed: {len(info_files)}")
    print(f"   - Total captions: {len(all_captions)}")
    print(f"   - Avg captions per file: {len(all_captions) / len(info_files):.1f}")
    
    return all_captions

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate,
    })


if __name__ == "__main__":
    main()
