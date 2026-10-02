from typing import List, Tuple, Dict, NamedTuple


class PatchInfo(NamedTuple):
    """Information about a patch and its intersection with the original bbox."""
    index: int
    intersection_ratio: float  # How much of the patch is covered by bbox
    local_bbox: Tuple[float, float, float, float]  # Bbox within this patch (normalized 0-1)

def convert_bbox_to_patch_indices_precise(bbox: List[int],
                                         num_patches_per_row: int,
                                         num_patches_per_col: int,
                                         image_size: Tuple[int, int],
                                         min_intersection_ratio: float = 0.1
                                         ) -> Tuple[List[int], Dict[int, PatchInfo]]:
    """
    Convert a bounding box to patch indices with precise intersection information.
    
    Args:
        bbox: List of [x, y, w, h] coordinates
        num_patches_per_row: Number of patches horizontally
        num_patches_per_col: Number of patches vertically
        image_size: Tuple of (width, height) of the original image
        min_intersection_ratio: Minimum intersection ratio to include a patch
    
    Returns:
        Tuple of (patch_indices, patch_info_dict)
    """
    x, y, w, h = bbox
    img_width, img_height = image_size
    
    # Calculate patch dimensions
    patch_width = img_width / num_patches_per_row
    patch_height = img_height / num_patches_per_col
    
    # Find potentially intersecting patches with larger bounds
    col_start = max(0, int((x - patch_width) // patch_width))
    col_end = min(num_patches_per_row - 1, int((x + w + patch_width) // patch_width))
    row_start = max(0, int((y - patch_height) // patch_height))
    row_end = min(num_patches_per_col - 1, int((y + h + patch_height) // patch_height))
    
    patch_indices = []
    patch_info = {}
    
    # Check each potential patch for actual intersection
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            # Calculate patch boundaries
            patch_x = col * patch_width
            patch_y = row * patch_height
            patch_x2 = patch_x + patch_width
            patch_y2 = patch_y + patch_height
            
            # Calculate intersection
            intersect_x1 = max(x, patch_x)
            intersect_y1 = max(y, patch_y)
            intersect_x2 = min(x + w, patch_x2)
            intersect_y2 = min(y + h, patch_y2)
            
            # Check if there's actual intersection
            if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                # Calculate intersection area and ratio
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                patch_area = patch_width * patch_height
                intersection_ratio = intersect_area / patch_area
                
                # Only include if intersection ratio is above threshold
                if intersection_ratio >= min_intersection_ratio:
                    patch_index = row * num_patches_per_row + col
                    
                    # Calculate local bbox within patch (normalized 0-1)
                    local_x1 = (intersect_x1 - patch_x) / patch_width
                    local_y1 = (intersect_y1 - patch_y) / patch_height
                    local_x2 = (intersect_x2 - patch_x) / patch_width
                    local_y2 = (intersect_y2 - patch_y) / patch_height
                    
                    patch_info[patch_index] = PatchInfo(
                        index=patch_index,
                        intersection_ratio=intersection_ratio,
                        local_bbox=(local_x1, local_y1, local_x2, local_y2)
                    )
                    patch_indices.append(patch_index)
    
    return patch_indices, patch_info


def convert_patch_indices_to_bbox_xywh_precise(patch_info: Dict[int, PatchInfo],
                                              num_patches_per_row: int,
                                              num_patches_per_col: int,
                                              image_size: Tuple[int, int]
                                              ) -> List[int]:
    """
    Convert patch info back to precise bounding box using intersection information.
    
    Args:
        patch_info: Dictionary mapping patch indices to PatchInfo
        num_patches_per_row: Number of patches horizontally
        num_patches_per_col: Number of patches vertically
        image_size: Tuple of (width, height) of the original image
    
    Returns:
        Reconstructed bounding box as [x, y, w, h] list of integers
    """
    if not patch_info:
        return [0, 0, 0, 0]
    
    img_width, img_height = image_size
    patch_width = img_width / num_patches_per_row
    patch_height = img_height / num_patches_per_col
    
    # Collect all intersection boundaries
    x_coords = []
    y_coords = []
    
    for info in patch_info.values():
        row = info.index // num_patches_per_row
        col = info.index % num_patches_per_row
        
        # Convert local bbox back to global coordinates
        patch_x = col * patch_width
        patch_y = row * patch_height
        
        local_x1, local_y1, local_x2, local_y2 = info.local_bbox
        
        global_x1 = patch_x + local_x1 * patch_width
        global_y1 = patch_y + local_y1 * patch_height
        global_x2 = patch_x + local_x2 * patch_width
        global_y2 = patch_y + local_y2 * patch_height
        
        x_coords.extend([global_x1, global_x2])
        y_coords.extend([global_y1, global_y2])
    
    # Find bounding box of all intersections
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # Convert to integer bbox
    x = int(round(x_min))
    y = int(round(y_min))
    w = int(round(x_max - x_min))
    h = int(round(y_max - y_min))
    
    # Ensure bbox doesn't exceed image boundaries
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    return [x, y, w, h]


# Fallback functions for simple use cases (original functions)
def convert_bbox_to_patch_indices(bbox: List[int],
                                  num_patches_per_row: int,
                                  num_patches_per_col: int,
                                  image_size: Tuple[int, int],
                                  min_intersection_ratio: float = 0.0
                                  ) -> List[int]:
    """Simple version that returns only patch indices."""
    indices, _ = convert_bbox_to_patch_indices_precise(
        bbox, num_patches_per_row, num_patches_per_col, image_size, min_intersection_ratio
    )
    return indices


def convert_patch_indices_to_bbox_xywh(patch_indices: List[int],
                                       num_patches_per_row: int,
                                       num_patches_per_col: int,
                                       image_size: Tuple[int, int]
                                       ) -> List[int]:
    """Simple version that reconstructs bbox from patch indices only (less precise)."""
    if not patch_indices:
        return [0, 0, 0, 0]
    
    img_width, img_height = image_size
    patch_width = img_width / num_patches_per_row
    patch_height = img_height / num_patches_per_col
    
    rows = [idx // num_patches_per_row for idx in patch_indices]
    cols = [idx % num_patches_per_row for idx in patch_indices]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    x = int(min_col * patch_width)
    y = int(min_row * patch_height)
    w = int((max_col + 1) * patch_width) - x
    h = int((max_row + 1) * patch_height) - y
    
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    return [x, y, w, h]


# Advanced reconstruction using weighted centroid approach
def convert_patch_indices_to_bbox_xywh_weighted(patch_info: Dict[int, PatchInfo],
                                               num_patches_per_row: int,
                                               num_patches_per_col: int,
                                               image_size: Tuple[int, int]
                                               ) -> List[int]:
    """
    Reconstruct bbox using weighted centroid of intersections for better precision.
    """
    if not patch_info:
        return [0, 0, 0, 0]
    
    img_width, img_height = image_size
    patch_width = img_width / num_patches_per_row
    patch_height = img_height / num_patches_per_col
    
    # Calculate weighted boundaries
    weighted_x_min = 0
    weighted_y_min = 0
    weighted_x_max = 0
    weighted_y_max = 0
    total_weight = 0
    
    for info in patch_info.values():
        row = info.index // num_patches_per_row
        col = info.index % num_patches_per_row
        
        patch_x = col * patch_width
        patch_y = row * patch_height
        
        local_x1, local_y1, local_x2, local_y2 = info.local_bbox
        weight = info.intersection_ratio
        
        global_x1 = patch_x + local_x1 * patch_width
        global_y1 = patch_y + local_y1 * patch_height
        global_x2 = patch_x + local_x2 * patch_width
        global_y2 = patch_y + local_y2 * patch_height
        
        weighted_x_min += global_x1 * weight
        weighted_y_min += global_y1 * weight
        weighted_x_max += global_x2 * weight
        weighted_y_max += global_y2 * weight
        total_weight += weight
    
    if total_weight > 0:
        x_min = weighted_x_min / total_weight
        y_min = weighted_y_min / total_weight
        x_max = weighted_x_max / total_weight
        y_max = weighted_y_max / total_weight
    else:
        return convert_patch_indices_to_bbox_xywh_precise(patch_info, num_patches_per_row, num_patches_per_col, image_size)
    
    x = int(round(x_min))
    y = int(round(y_min))
    w = int(round(x_max - x_min))
    h = int(round(y_max - y_min))
    
    # Clamp to image boundaries
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    return [x, y, w, h]


# Example usage and comparison
if __name__ == "__main__":
    # Test precision improvements
    bbox_original = [35, 45, 87, 63]
    image_size = (224, 224)
    
    print("=== Precision Comparison ===")
    print(f"Original bbox: {bbox_original}")
    
    # Method 1: Simple (original)
    patches_simple = convert_bbox_to_patch_indices(bbox_original, 14, 14, image_size)
    reconstructed_simple = convert_patch_indices_to_bbox_xywh(patches_simple, 14, 14, image_size)
    
    # Method 2: Precise intersection info
    patches_precise, patch_info = convert_bbox_to_patch_indices_precise(bbox_original, 14, 14, image_size)
    reconstructed_precise = convert_patch_indices_to_bbox_xywh_precise(patch_info, 14, 14, image_size)
    
    # Method 3: Weighted reconstruction
    reconstructed_weighted = convert_patch_indices_to_bbox_xywh_weighted(patch_info, 14, 14, image_size)
    
    print(f"Simple method:    {reconstructed_simple} (patches: {len(patches_simple)})")
    print(f"Precise method:   {reconstructed_precise} (patches: {len(patches_precise)})")
    print(f"Weighted method:  {reconstructed_weighted}")
    
    # Calculate errors
    def bbox_error(original, reconstructed):
        return [abs(o - r) for o, r in zip(original, reconstructed)]
    
    error_simple = bbox_error(bbox_original, reconstructed_simple)
    error_precise = bbox_error(bbox_original, reconstructed_precise)
    error_weighted = bbox_error(bbox_original, reconstructed_weighted)
    
    print(f"\nErrors [x, y, w, h]:")
    print(f"Simple:    {error_simple} (total: {sum(error_simple)})")
    print(f"Precise:   {error_precise} (total: {sum(error_precise)})")
    print(f"Weighted:  {error_weighted} (total: {sum(error_weighted)})")
    
    # Show patch info details
    print(f"\nPatch intersection details:")
    for idx, info in list(patch_info.items())[:3]:  # Show first 3
        print(f"  Patch {idx}: ratio={info.intersection_ratio:.3f}, local_bbox={[f'{x:.3f}' for x in info.local_bbox]}")