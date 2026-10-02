import json
import re
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from torchvision.datasets import CocoDetection
from composer.data.datasets.utils import rescale_bbox, reconstruct_patch_info_from_ground_truth, convert_patch_indices_to_bbox_xywh


class RECBboxParser:
    def __init__(self, ann_file, img_prefix):
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
        bbox_pattern = r'<bbox>.*?x=(\d+).*?y=(\d+).*?w=(\d+).*?h=(\d+).*?</bbox>'
        bbox_match = re.search(bbox_pattern, match, re.DOTALL)
        if bbox_match:
            coords = [float(x) for x in bbox_match.groups()]

            instance_data['bbox'] = coords
        
        return instance_data if instance_data else None
    
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
            comparison['label_match'] = True
            # IoU metrics for empty case
            iou_metrics = {f'iou@{k}': 1.0 for k in iou_thresholds}
            iou_metrics['raw_iou'] = 1.0
            comparison['iou_metrics'] = iou_metrics
            return comparison
        
        if not pred_instance or not target_instance:
            comparison['label_match'] = False
            comparison['pred_label'] = pred_instance.get('label', '') if pred_instance else ''
            comparison['target_label'] = target_instance.get('label', '') if target_instance else ''
            # IoU metrics for missing instance case
            iou_metrics = {f'iou@{k}': 0.0 for k in iou_thresholds}
            iou_metrics['raw_iou'] = 0.0
            comparison['iou_metrics'] = iou_metrics
            return comparison
        
        # Both instances exist - compare them
        pred_label = pred_instance.get('label', '')
        target_label = target_instance.get('label', '')


        comparison['pred_label'] = pred_label
        comparison['target_label'] = target_label
        comparison['label_match'] = pred_label == target_label
                
        pred_bbox_xywh = pred_instance.get('bbox')
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
    parser = RECBboxParser(ann_file="dataset/refcoco/ann/refcoco_train.json", img_prefix="dataset/coco/train2017")
    
    file_path = "/home/thodemon/workspace/composer-rebuild/eval_rec/eval_rec/refcoco_testA/377227/checkpoint-final_0.jsonl"

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