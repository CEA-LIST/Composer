from typing import List, Tuple, Dict, NamedTuple
from composer.constants import SPECIAL_TOKENS, PROXY_TOKENS




class PatchInfo(NamedTuple):
    """Information about a patch and its intersection with the original bbox."""
    index: int
    intersection_ratio: float  # How much of the patch is covered by bbox
    local_bbox: Tuple[float, float, float, float]  # Bbox within this patch (normalized 0-1)


def reconstruct_patch_info_from_ground_truth(patch_indices: List[int],
                                           ground_truth_bbox: List[int],
                                           num_patches_per_row: int = 16,
                                           num_patches_per_col: int = 16,
                                           image_size: Tuple[int, int] = (224, 224)
                                           ) -> Dict[int, PatchInfo]:
    """
    Reconstruct patch_info dictionary using ground truth bbox and patch indices.
    This allows you to use the original convert_patch_indices_to_bbox_xywh function.
    
    Args:
        patch_indices: List of patch indices
        ground_truth_bbox: Ground truth bounding box [x, y, w, h]
        num_patches_per_row: Number of patches horizontally
        num_patches_per_col: Number of patches vertically
        image_size: Tuple of (width, height) of the original image

    Returns:
        Dictionary mapping patch indices to PatchInfo objects
    """
    x, y, w, h = ground_truth_bbox
    img_width, img_height = image_size
    
    patch_width = img_width / num_patches_per_row
    patch_height = img_height / num_patches_per_col
    
    patch_info = {}
    
    for patch_idx in patch_indices:
        row = patch_idx // num_patches_per_row
        col = patch_idx % num_patches_per_row
        
        # Calculate patch boundaries
        patch_x = col * patch_width
        patch_y = row * patch_height
        patch_x2 = patch_x + patch_width
        patch_y2 = patch_y + patch_height
        
        # Calculate intersection with ground truth bbox
        intersect_x1 = max(x, patch_x)
        intersect_y1 = max(y, patch_y)
        intersect_x2 = min(x + w, patch_x2)
        intersect_y2 = min(y + h, patch_y2)
        
        # Calculate intersection area and ratio
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        patch_area = patch_width * patch_height
        intersection_ratio = intersect_area / patch_area
        
        # Calculate local bbox within patch (normalized 0-1)
        local_x1 = (intersect_x1 - patch_x) / patch_width
        local_y1 = (intersect_y1 - patch_y) / patch_height
        local_x2 = (intersect_x2 - patch_x) / patch_width
        local_y2 = (intersect_y2 - patch_y) / patch_height
        
        patch_info[patch_idx] = PatchInfo(
            index=patch_idx,
            intersection_ratio=intersection_ratio,
            local_bbox=(local_x1, local_y1, local_x2, local_y2)
        )
    
    return patch_info


def convert_bbox_to_patch_indices(bbox: List[int],
                                 num_patches_per_row: int = 16,
                                 num_patches_per_col: int = 16,
                                 image_size: Tuple[int, int] = (224, 224),
                                 min_intersection_ratio: float = 0.1
                                 ) -> Tuple[List[int], Dict[int, PatchInfo]]:
    """
    Convert a bounding box ([x, y, w, h] format) to a sequence of patch indices 
    with precise intersection information.

        +----+----+----+----+
        |  0 |  1 |  2 | .. |   ← Row 0
        +----+----+----+----+
        | 16 | 17 | 18 | .. |   ← Row 1
        +----+----+----+----+
        | 33 | .. | .. | .. |   ← Row 2
        +----+----+----+----+
        | .. | .. | .. | .. |   ← Row 15
        +----+----+----+----+

    Args:
        bbox: List of [x, y, w, h] coordinates
        num_patches_per_row: Number of patches horizontally
        num_patches_per_col: Number of patches vertically
        image_size: Tuple of (width, height) of the original image
        min_intersection_ratio: Minimum intersection ratio to include a patch

    Returns:
        Tuple of (patch_indices, patch_info_dict) where patch_info_dict contains
        precise intersection information for each patch.

    Note: 
        Bounding boxes are in [x, y, w, h] format
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


def convert_patch_indices_to_bbox_xywh(patch_info: Dict[int, PatchInfo],
                                      num_patches_per_row: int = 16,
                                      num_patches_per_col: int = 16,
                                      image_size: Tuple[int, int] = (224, 224)
                                      ) -> List[int]:
    """
    Convert patch info back to precise bounding box using intersection information.
    
    Args:
        patch_info: Dictionary mapping patch indices to PatchInfo
        num_patches_per_row: Number of patches horizontally
        num_patches_per_col: Number of patches vertically
        image_size: Tuple of (width, height) of the original image

    Returns:
        Reconstructed bounding box as [x, y, w, h] list of integers.
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


def convert_bbox_to_region_tokens(
        bbox: List[int],
        num_patches_per_row: int = 16,
        num_patches_per_col: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        return_patch_info_dict: bool = False
        ) -> str:

    """
    Convert a bounding box ([x, y, w, h] format) to a sequence of region tokens (e.g., <r1><r2><r4>).

    Args:
        bbox: List of [x, y, w, h] coordinates.

    Returns:
        The sequence of region tokens as a string.

    Note: 
        Bounding boxes are in [x, y, w, h] format
    """
    
    indices, _ = convert_bbox_to_patch_indices(bbox, num_patches_per_row, num_patches_per_col, image_size)

    # if the bounding box is empty, return the <no_region> token
    if len(indices) == 0:
        return SPECIAL_TOKENS["NO_REGION"]
     
    region_tokens = []
    for patch_idx in indices:
        region_token = PROXY_TOKENS[patch_idx]
        region_tokens.append(region_token)
    
    return "".join(region_tokens)


def rescale_bbox(bbox: List[int], 
                 x_factor: float, 
                 y_factor: float
                 ) -> List[int]:
    """Rescale a bounding box to the new image size.
    
    Args:
        bbox: List of [x, y, w, h] coordinates.
        x_factor: Factor to rescale the x-coordinates.
        y_factor: Factor to rescale the y-coordinates.

    Returns:
        List of [x, y, w, h] coordinates.
    """

    return [int(bbox[0] * x_factor), 
            int(bbox[1] * y_factor), 
            int(bbox[2] * x_factor), 
            int(bbox[3] * y_factor)]




def convert_region_token_ids_to_patch_indices(region_token_ids: List[int],
                                              token_offset: int
                                              ) -> List[int]:
    """
    Convert a list of token IDs (corresponding to region tokens) to a list of patch indices 
    considering the following grid:

         +----+----+----+----+
        |  0 |  1 |  2 | .. |   ← Row 0
        +----+----+----+----+
        | 16 | 17 | 18 | .. |   ← Row 1
        +----+----+----+----+
        | 33 | .. | .. | .. |   ← Row 2
        +----+----+----+----+
        | .. | .. | .. | .. |   ← Row 15
        +----+----+----+----+
    
    Args:
        region_token_ids: List of region token IDs.
        token_offset: The offset of the region tokens obtained from the tokenizer.

    Returns:
        The list of patch indices.
    """

    return [region_token_id - token_offset for region_token_id in region_token_ids]


def convert_region_tokens_to_bbox(region_token_ids: List[int],
                                  token_offset: int,
                                  grid_size: int = 16
                                  ) -> List[int]:
        """
        Convert a list of token IDs (corresponding to region tokens) to a bounding box in [x, y, w, h] format.

        Args:
            region_token_ids: List of region token IDs.
            token_offset: The offset of the region tokens obtained from the tokenizer.
            grid_size: The size of a grid unit (e.g., 16 for a 16x16 grid).

        Returns:
            The bounding box in [x, y, w, h] format.
        """
        if not region_token_ids:
            return (0.0, 0.0, 0.0, 0.0)
        
        patch_indices = convert_region_token_ids_to_patch_indices(region_token_ids, token_offset)

        return convert_patch_indices_to_bbox_xywh(patch_indices, grid_size)