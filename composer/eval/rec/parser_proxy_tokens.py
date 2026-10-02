import json
import re
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from torchvision.datasets import CocoDetection
from composer.data.datasets.utils import rescale_bbox, reconstruct_patch_info_from_ground_truth, convert_patch_indices_to_bbox_xywh


class RECProxyTokensParser:
    def __init__(self, ann_file, img_prefix, grid_size: int = 16):
        self.grid_size = grid_size
        self.coco_dataset = CocoDetection(img_prefix, ann_file)
        self.coco = self.coco_dataset.coco
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1, bbox2: Bounding boxes in [x, y, w, h] format
            
        Returns:
            IoU score between 0 and 1
        """
        if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to [x1, y1, x2, y2] format for easier calculation
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return float(intersection_area) / union_area
    
    def calculate_iou_at_k_tiles(self, pred_indices, target_indices, k_values: List[float] = [0.5, 0.75, 0.9]):
        if not pred_indices or not target_indices:
            iou_metrics = {f'iou@{k}': 0.0 for k in k_values}
            iou_metrics['raw_iou'] = 0.0
            return iou_metrics
        
        def calculate_iou_tiles_level(pred, target):
            """
            Calculate Intersection over Union (IoU) for two sets of tile indices.
            
            Parameters:
            - pred: list of predicted tile indices
            - target: list of target tile indices
            
            Returns:
            - iou: IoU score (0.0 to 1.0)
            - intersection: number of overlapping tiles
            - union: total number of unique tiles
            """
            pred_set = set(pred)
            target_set = set(target)
            
            intersection = len(pred_set & target_set)
            union = len(pred_set | target_set)
            
            if union == 0:
                return 0.0, 0, 0
            
            iou = intersection / union
            return iou
        
        # Calculate IoU
        iou_score = calculate_iou_tiles_level(pred_indices, target_indices)
        
        # Calculate IoU@k for different thresholds
        iou_metrics = {'raw_iou': iou_score}
        for k in k_values:
            iou_metrics[f'iou@{k}'] = 1.0 if iou_score >= k else 0.0
        
        return iou_metrics
    
    def calculate_iou_at_k(self, 
                           pred_bbox: List[int], 
                           target_bbox: List[int], 
                           k_values: List[float] = [0.5, 0.75, 0.9]) -> Dict[str, float]:
        """Calculate IoU@k metrics for different IoU thresholds.
        
        Args:
            pred_bbox: Predicted bbox
            target_bbox: Target bbox
            k_values: IoU thresholds to evaluate
            
        Returns:
            Dictionary with IoU@k scores and raw IoU value
        """
        if not pred_bbox or not target_bbox:
            iou_metrics = {f'iou@{k}': 0.0 for k in k_values}
            iou_metrics['raw_iou'] = 0.0
            return iou_metrics
        
        # Calculate IoU
        iou_score = self.calculate_iou(pred_bbox, target_bbox)
        
        # Calculate IoU@k for different thresholds
        iou_metrics = {'raw_iou': iou_score}
        for k in k_values:
            iou_metrics[f'iou@{k}'] = 1.0 if iou_score >= k else 0.0
        
        return iou_metrics
    
    def extract_reasoning_and_answer(self, text: str) -> Dict[str, str]:
        """Extract reasoning and answer sections from the text - only considers the FIRST reasoning block."""
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        reasoning_matches = re.findall(reasoning_pattern, text, re.DOTALL)
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        
        return {
            'reasoning': reasoning_matches[0].strip() if reasoning_matches else '',
            'answer': answer_matches[0].strip() if answer_matches else ''
        }
    
    def extract_object_recognition(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract single objectInstance from the FIRST reasoning block."""
        # First, extract only the FIRST reasoning block
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        reasoning_matches = re.findall(reasoning_pattern, text, re.DOTALL)
        
        if not reasoning_matches:
            return None
        
        # Use only the first reasoning block
        first_reasoning = reasoning_matches[0]
        
        # Find the first objectInstance block within the first reasoning block
        object_pattern = r'<objectInstance>(.*?)</objectInstance>'
        object_match = re.search(object_pattern, first_reasoning, re.DOTALL)
        
        if not object_match:
            return None
        
        instance_data = {}
        match = object_match.group(1)
        
        # Extract label
        label_pattern = r'<label>(.*?)</label>'
        label_match = re.search(label_pattern, match)
        if label_match:
            instance_data['label'] = label_match.group(1).strip()
        
        # Extract bbox sequence
        bbox_pattern = r'<bbox>(.*?)</bbox>'
        bbox_match = re.search(bbox_pattern, match, re.DOTALL)
        if bbox_match:
            bbox_content = bbox_match.group(1)
            # Extract all r-tags (like <r72>, <r73>, etc.)
            r_pattern = r'<r(\d+)>'
            r_numbers = re.findall(r_pattern, bbox_content)
            instance_data['patch_indices'] = [int(num)-1 for num in r_numbers] # patch indices are from 0 to 255
        
        return instance_data if instance_data else None
    
    def calculate_patch_indices_score(self, pred_sequence: List[int], target_sequence: List[int]) -> Dict[str, float]:
        """Calculate various scores comparing predicted and target bbox sequences."""
        if not pred_sequence and not target_sequence:
            return {'exact_match': 1.0, 'jaccard': 1.0, 'sequence_similarity': 1.0, 'order_score': 1.0}
        
        if not pred_sequence or not target_sequence:
            return {'exact_match': 0.0, 'jaccard': 0.0, 'sequence_similarity': 0.0, 'order_score': 0.0}
        
        # Exact match score
        exact_match = 1.0 if pred_sequence == target_sequence else 0.0
        
        # Jaccard similarity (intersection over union)
        pred_set = set(pred_sequence)
        target_set = set(target_sequence)
        intersection = len(pred_set.intersection(target_set))
        union = len(pred_set.union(target_set))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Sequence similarity using difflib
        sequence_similarity = SequenceMatcher(None, pred_sequence, target_sequence).ratio()
        
        # Order preservation score (how well the relative order is maintained)
        common_elements = list(pred_set.intersection(target_set))
        if len(common_elements) <= 1:
            order_score = 1.0 if common_elements else 0.0
        else:
            # Check if common elements maintain their relative order
            pred_positions = {val: i for i, val in enumerate(pred_sequence) if val in common_elements}
            target_positions = {val: i for i, val in enumerate(target_sequence) if val in common_elements}
            
            order_violations = 0
            for i, elem1 in enumerate(common_elements):
                for j, elem2 in enumerate(common_elements[i+1:], i+1):
                    pred_order = pred_positions[elem1] < pred_positions[elem2]
                    target_order = target_positions[elem1] < target_positions[elem2]
                    if pred_order != target_order:
                        order_violations += 1
            
            total_pairs = len(common_elements) * (len(common_elements) - 1) // 2
            order_score = 1.0 - (order_violations / total_pairs) if total_pairs > 0 else 1.0
        
        return {
            'exact_match': exact_match,
            'jaccard': jaccard,
            'sequence_similarity': sequence_similarity,
            'order_score': order_score
        }
    
    def compare_object_recognition(self, 
                                   pred_instance: Optional[Dict], 
                                   target_instance: Optional[Dict],
                                   iou_thresholds: List[float] = [0.5, 0.75, 0.9]
                                   ) -> Dict[str, Any]:
        
        """Compare predicted and target object recognition instances."""
        comparison = {
            'has_pred_instance': pred_instance is not None,
            'has_target_instance': target_instance is not None,
        }
        
        if not pred_instance and not target_instance:
            comparison['scores'] = {'exact_match': 1.0, 'jaccard': 1.0, 'sequence_similarity': 1.0, 'order_score': 1.0}
            comparison['label_match'] = True
            # IoU metrics for empty case
            iou_metrics = {f'iou@{k}': 1.0 for k in iou_thresholds}
            iou_metrics['raw_iou'] = 1.0
            comparison['iou_metrics'] = iou_metrics
            return comparison
        
        if not pred_instance or not target_instance:
            comparison['scores'] = {'exact_match': 0.0, 'jaccard': 0.0, 'sequence_similarity': 0.0, 'order_score': 0.0}
            comparison['label_match'] = False
            comparison['pred_label'] = pred_instance.get('label', '') if pred_instance else ''
            comparison['target_label'] = target_instance.get('label', '') if target_instance else ''
            comparison['pred_bbox_length'] = len(pred_instance.get('patch_indices', [])) if pred_instance else 0
            comparison['target_bbox_length'] = len(target_instance.get('patch_indices', [])) if target_instance else 0
            # IoU metrics for missing instance case
            iou_metrics = {f'iou@{k}': 0.0 for k in iou_thresholds}
            iou_metrics['raw_iou'] = 0.0
            comparison['iou_metrics'] = iou_metrics
            return comparison
        
        # Both instances exist - compare them
        pred_label = pred_instance.get('label', '')
        target_label = target_instance.get('label', '')
        pred_patch_indices = pred_instance.get('patch_indices', [])
        target_patch_indices = target_instance.get('patch_indices', [])
        

        comparison['pred_label'] = pred_label
        comparison['target_label'] = target_label
        comparison['pred_bbox_length'] = len(pred_patch_indices)
        comparison['target_bbox_length'] = len(target_patch_indices)
        comparison['label_match'] = pred_label == target_label
        
        # Calculate bbox sequence scores
        comparison['scores'] = self.calculate_patch_indices_score(pred_patch_indices, target_patch_indices)
        
        pred_bbox_xywh = convert_patch_indices_to_bbox_xywh(reconstruct_patch_info_from_ground_truth(pred_patch_indices, target_instance.get('gt_bbox_rescaled')))
        target_bbox_xywh = target_instance.get('gt_bbox_rescaled')
        
        comparison['iou_metrics'] = self.calculate_iou_at_k(pred_bbox_xywh, target_bbox_xywh, iou_thresholds)

        # Add converted bounding boxes for debugging
        comparison['pred_bbox_xywh'] = pred_bbox_xywh
        comparison['target_bbox_xywh'] = target_bbox_xywh
        
        return comparison

    def parse_jsonl_line(self, line: str) -> Dict[str, Any]:
        """Parse a single line from the JSONL file."""
        try:
            data = json.loads(line.strip())
            prediction = data.get('prediction', '')
            target_sequence = data.get('target_sequence', '')
            
            width = data.get('width', None)
            height = data.get('height', None)

            # Extract reasoning and answer from prediction
            pred_reasoning_answer = self.extract_reasoning_and_answer(prediction)
            pred_object_instance = self.extract_object_recognition(prediction)
            
            # Extract reasoning and answer from target
            target_reasoning_answer = self.extract_reasoning_and_answer(target_sequence)
            target_object_instance = self.extract_object_recognition(target_sequence)

            target_object_instance["gt_bbox_rescaled"] = rescale_bbox(data.get('gt_bbox', []), 224/width, 224/height)

            # Compare instances
            comparison = self.compare_object_recognition(pred_object_instance, target_object_instance)
            
            result = {
                'id': data.get('id'),
                'image_file': data.get('image_file'),
                'gt_bboxes': data.get('gt_bbox'),
                'prediction': {
                    'reasoning': pred_reasoning_answer['reasoning'],
                    'answer': pred_reasoning_answer['answer'],
                    'object_instance': pred_object_instance
                },
                'target': {
                    'reasoning': target_reasoning_answer['reasoning'],
                    'answer': target_reasoning_answer['answer'],
                    'object_instance': target_object_instance
                },
                'comparison': comparison
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
        except Exception as e:
            print(f"Error processing line: {e}")
            return None
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse an entire JSONL file."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():  # Skip empty lines
                        parsed_line = self.parse_jsonl_line(line)
                        if parsed_line:
                            results.append(parsed_line)
                        else:
                            print(f"Failed to parse line {line_num}")
            
            return results
            
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
    
    def calculate_dataset_metrics(self, 
                                  results: List[Dict[str, Any]]
                                  ) -> Dict[str, float]:
        
        """Calculate average metrics across the entire dataset."""
        if not results:
            return {}
        
        metrics = {}
        
        # Sequence-based metrics
        sequence_metrics = ['exact_match', 'jaccard', 'sequence_similarity', 'order_score']
        for metric in sequence_metrics:
            values = [r['comparison']['scores'][metric] for r in results if 'scores' in r['comparison']]
            metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0.0
        
        # IoU metrics
        iou_keys = ['raw_iou', 'iou@0.5', 'iou@0.75', 'iou@0.9']  # Adjust based on your thresholds
        for iou_key in iou_keys:
            values = [r['comparison']['iou_metrics'][iou_key] for r in results 
                     if 'iou_metrics' in r['comparison'] and iou_key in r['comparison']['iou_metrics']]
            metrics[f'avg_{iou_key}'] = sum(values) / len(values) if values else 0.0

        # Label accuracy
        label_matches = [r['comparison']['label_match'] for r in results 
                        if 'label_match' in r['comparison']]
        metrics['label_accuracy'] = sum(label_matches) / len(label_matches) if label_matches else 0.0
        
        return metrics

def main():
    parser = RECProxyTokensParser(ann_file="dataset/refcoco/ann/refcoco_val.json",
                                     img_prefix="dataset/coco/train2017",
                                     grid_size=16)
    

    file_path = "/home/thodemon/workspace/composer-rebuild/eval_rec/eval_rec/refcoco_val/377226/checkpoint-final_0.jsonl"

    results = parser.parse(file_path)

    dataset_metrics = parser.calculate_dataset_metrics(results)
    print(dataset_metrics)
    print("\n=== DATASET METRICS ===")
    for metric_name, value in dataset_metrics.items():
        print(f"{metric_name}: {value:.3f}")


    # c = set()
    # l = 20
    # count = 0
    # for result in results:
        
    #     score = result["comparison"].get("iou_metrics", None)
    #     if score is not None:
    #         if score["iou@0.5"] == 0.0:
    #             count+=1
    #             print(result)
    #             print(result["prediction"]["object_instance"]["patch_indices"])
    #             print(result["target"]["object_instance"]["patch_indices"])
    #             print('\n \n')
    #             c.add(tuple(result['gt_bboxes']))
                
    #             if count >= l:
    #                 break

    # print(len(c))

if __name__ == "__main__":
    main()