import json
import re
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
from composer.data.datasets.utils import rescale_bbox, reconstruct_patch_info_from_ground_truth, convert_patch_indices_to_bbox_xywh


class HeuristicbaseBboxParser:
    def __init__(self, ann_file: str, img_prefix: str, grid_size: int = 16):
        self.grid_size = grid_size
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        
        # Define task types and their expected structure
        self.task_types = {
            'objectRecognition': ['objectInstance'],
            'spatialPosition': ['spatialPositionInstance', 'objectInstance'],
            'attributeColor': ['objectInstance', 'attributeColorInstance'],
            'spatialRelationship': ['objectInstance', 'relationInstance', 'subjectInstance']
        }
    
        self.ann = self.load_annotation_data()

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
        
    def load_annotation_data(self) -> Dict[str, Any]:
        """Load annotation data from JSONL file."""
        annotations = {}
        try:
            with open(self.ann_file, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        data = json.loads(line.strip())
                        key = data.get('question_id')
                        if key:
                            annotations[key] = data
        except FileNotFoundError:
            print(f"Annotation file {self.ann_file} not found")
        except Exception as e:
            print(f"Error loading annotations: {e}")
        
        return annotations
    
    def extract_reasoning_and_answer(self, text: str) -> Dict[str, str]:
        """Extract reasoning and answer sections from the text."""
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        reasoning_matches = re.findall(reasoning_pattern, text, re.DOTALL)
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        
        return {
            'reasoning': reasoning_matches[0].strip() if reasoning_matches else '',
            'answer': answer_matches[0].strip() if answer_matches else ''
        }
    
    def extract_object_instance(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract object instance information from text."""
        object_pattern = r'<objectInstance>(.*?)</objectInstance>'
        object_match = re.search(object_pattern, text, re.DOTALL)
        
        if not object_match:
            return None
        
        instance_data = {}
        match_content = object_match.group(1)
        
        # Extract label
        label_pattern = r'<label>(.*?)</label>'
        label_match = re.search(label_pattern, match_content)
        if label_match:
            instance_data['label'] = label_match.group(1).strip()
        
        # Extract bbox sequence
        bbox_pattern = r'<bbox>.*?x=(\d+).*?y=(\d+).*?w=(\d+).*?h=(\d+).*?</bbox>'
        bbox_match = re.search(bbox_pattern, match_content, re.DOTALL)
        if bbox_match:
            coords = [float(x) for x in bbox_match.groups()]

            instance_data['bbox'] = coords
        
        return instance_data if instance_data else None
    
    def extract_spatial_position_instance(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract spatial position instance information from text."""
        spatial_pattern = r'<spatialPositionInstance>(.*?)</spatialPositionInstance>'
        spatial_match = re.search(spatial_pattern, text, re.DOTALL)
        
        if not spatial_match:
            return None
        
        instance_data = {}
        match_content = spatial_match.group(1)
        
        # Extract spatial position
        position_pattern = r'<spatialPosition>(.*?)</spatialPosition>'
        position_match = re.search(position_pattern, match_content)
        if position_match:
            instance_data['spatial_position'] = position_match.group(1).strip()
        
        # Extract bbox sequence
        bbox_pattern = r'<bbox>.*?x=(\d+).*?y=(\d+).*?w=(\d+).*?h=(\d+).*?</bbox>'
        bbox_match = re.search(bbox_pattern, match_content, re.DOTALL)
        if bbox_match:
            coords = [float(x) for x in bbox_match.groups()]

            instance_data['bbox'] = coords
        
        return instance_data if instance_data else None
    
    def extract_attribute_color_instance(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract attribute color instance information from text."""
        color_pattern = r'<attributeColorInstance>(.*?)</attributeColorInstance>'
        color_match = re.search(color_pattern, text, re.DOTALL)
        
        if not color_match:
            return None
        
        instance_data = {}
        match_content = color_match.group(1)
        
        # Extract color
        color_value_pattern = r'<color>(.*?)</color>'
        color_value_match = re.search(color_value_pattern, match_content)
        if color_value_match:
            instance_data['color'] = color_value_match.group(1).strip()
        
        return instance_data if instance_data else None
    
    def extract_relation_instance(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract relation instance information from text."""
        relation_pattern = r'<relationInstance>(.*?)</relationInstance>'
        relation_match = re.search(relation_pattern, text, re.DOTALL)
        
        if not relation_match:
            return None
        
        instance_data = {}
        match_content = relation_match.group(1)
        
        # Extract relation
        relation_value_pattern = r'<relation>(.*?)</relation>'
        relation_value_match = re.search(relation_value_pattern, match_content)
        if relation_value_match:
            instance_data['relation'] = relation_value_match.group(1).strip()
        
        return instance_data if instance_data else None
    
    def extract_subject_instance(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract subject instance information from text."""
        subject_pattern = r'<subjectInstance>(.*?)</subjectInstance>'
        subject_match = re.search(subject_pattern, text, re.DOTALL)
        
        if not subject_match:
            return None
        
        instance_data = {}
        match_content = subject_match.group(1)
        
        # Extract label
        label_pattern = r'<label>(.*?)</label>'
        label_match = re.search(label_pattern, match_content)
        if label_match:
            instance_data['label'] = label_match.group(1).strip()
        
        # Extract bbox sequence
        bbox_pattern = r'<bbox>.*?x=(\d+).*?y=(\d+).*?w=(\d+).*?h=(\d+).*?</bbox>'
        bbox_match = re.search(bbox_pattern, match_content, re.DOTALL)
        if bbox_match:
            coords = [float(x) for x in bbox_match.groups()]

            instance_data['bbox'] = coords

        return instance_data if instance_data else None
    
    def parse_task_sequence(self, reasoning_text: str) -> List[Dict[str, Any]]:
        """Parse the sequence of tasks from reasoning text."""
        tasks = []
        
        # Define task patterns and their extractors
        task_patterns = {
            'objectRecognition': (r'<objectRecognition>(.*?)</objectRecognition>', self.parse_object_recognition),
            'spatialPosition': (r'<spatialPosition>(.*?)</spatialPosition>', self.parse_spatial_position),
            'attributeColor': (r'<attributeColor>(.*?)</attributeColor>', self.parse_attribute_color),
            'spatialRelationship': (r'<spatialRelationship>(.*?)</spatialRelationship>', self.parse_spatial_relationship)
        }
        
        # Find all tasks in order
        for task_type, (pattern, parser_func) in task_patterns.items():
            matches = re.finditer(pattern, reasoning_text, re.DOTALL)
            for match in matches:
                task_content = match.group(1)
                task_data = parser_func(task_content)
                if task_data:
                    task_data['task_type'] = task_type
                    task_data['start_pos'] = match.start()
                    tasks.append(task_data)
        
        # Sort tasks by their position in the text to maintain order
        tasks.sort(key=lambda x: x['start_pos'])
        
        # Remove start_pos as it's only used for sorting
        for task in tasks:
            del task['start_pos']
        
        return tasks
    
    def parse_object_recognition(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse object recognition task content."""
        object_instance = self.extract_object_instance(content)
        if object_instance:
            return {'object_instance': object_instance}
        return None
    
    def parse_spatial_position(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse spatial position task content."""
        spatial_instance = self.extract_spatial_position_instance(content)
        object_instance = self.extract_object_instance(content)
        
        result = {}
        if spatial_instance:
            result['spatial_position_instance'] = spatial_instance
        if object_instance:
            result['object_instance'] = object_instance
        
        return result if result else None
    
    def parse_attribute_color(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse attribute color task content."""
        object_instance = self.extract_object_instance(content)
        color_instance = self.extract_attribute_color_instance(content)
        
        result = {}
        if object_instance:
            result['object_instance'] = object_instance
        if color_instance:
            result['attribute_color_instance'] = color_instance
        
        return result if result else None
    
    def parse_spatial_relationship(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse spatial relationship task content."""
        object_instance = self.extract_object_instance(content)
        relation_instance = self.extract_relation_instance(content)
        subject_instance = self.extract_subject_instance(content)
        
        result = {}
        if object_instance:
            result['object_instance'] = object_instance
        if relation_instance:
            result['relation_instance'] = relation_instance
        if subject_instance:
            result['subject_instance'] = subject_instance
        
        return result if result else None
    
    def compare_task_sequences(self, pred_tasks: List[Dict[str, Any]], target_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare predicted and target task sequences."""
        comparison = {
            'task_sequence_match': False,
            'task_count_match': len(pred_tasks) == len(target_tasks),
            'pred_task_types': [task['task_type'] for task in pred_tasks],
            'target_task_types': [task['task_type'] for task in target_tasks],
            'task_order_score': 0.0,
            'bbox_comparisons': []
        }
       
        # Check if task sequences match exactly
        pred_task_types = [task['task_type'] for task in pred_tasks]
        target_task_types = [task['task_type'] for task in target_tasks]
        comparison['task_sequence_match'] = pred_task_types == target_task_types
        
        # Calculate task order score using sequence similarity
        comparison['task_order_score'] = SequenceMatcher(None, pred_task_types, target_task_types).ratio()
        
        return comparison
    
    def extract_all_bboxes_from_tasks(self, tasks: List[Dict[str, Any]]) -> List[List[int]]:
        """Extract all bounding boxes from a list of tasks."""
        bboxes = []
        for task in tasks:
            # Check different possible locations for patch_indices
            if 'object_instance' in task and 'bbox' in task['object_instance']:
                bboxes.append(task['object_instance']['bbox'])
            if 'spatial_position_instance' in task and 'bbox' in task['spatial_position_instance']:
                bboxes.append(task['spatial_position_instance']['bbox'])
            if 'subject_instance' in task and 'bbox' in task['subject_instance']:
                bboxes.append(task['subject_instance']['bbox'])
        return bboxes
    
    def compare_sequences(self, 
                          pred_reasoning: str, 
                          target_reasoning: str, 
                          pred_answer: str, 
                          target_answer: str,
                          gt_bboxes: List[List[int]]
        ) -> Dict[str, Any]:

        """Compare predicted and target sequences comprehensively."""
        # Parse task sequences
        pred_tasks = self.parse_task_sequence(pred_reasoning)
        target_tasks = self.parse_task_sequence(target_reasoning)
        
        # Compare final answers
        answer_match = pred_answer.strip().lower() == target_answer.strip().lower()

        # Compare task sequences
        task_comparison = self.compare_task_sequences(pred_tasks, target_tasks)
               
        # Compare bbox
        pred_bboxes = self.extract_all_bboxes_from_tasks(pred_tasks)
        target_bboxes = gt_bboxes

        raw_iou = []
        for p, t in  zip(pred_bboxes, target_bboxes):
            raw_iou.append(self.calculate_iou(p, t))
        
        avg_raw_iou = sum(raw_iou) / len(raw_iou)
        
        bbox_iou_metrics = dict()
        bbox_iou_metrics["avg_raw_iou"] = avg_raw_iou
        return {
            'answer_match': answer_match,
            'pred_answer': pred_answer.strip(),
            'target_answer': target_answer.strip(),
            'task_comparison': task_comparison,
            'pred_tasks': pred_tasks,
            'target_tasks': target_tasks,
            'bbox_iou_metrics': bbox_iou_metrics
        }
    

    def get_gt_bbox(self, ann):
        # to return object
        bboxes = list()

        # get all bboxes
        for task in ann["atomic_tasks"]:
            if 'objectInstance' in task:
                bboxes.append(task['objectInstance']['bbox'])
            if 'spatialPositionInstance' in task:
                bboxes.append(task['spatialPositionInstance']['bbox'])
            if 'subjectInstance' in task:
                bboxes.append(task['subjectInstance']['bbox'])

        return bboxes
    
    def parse_jsonl_line(self, line: str) -> Dict[str, Any]:
        """Parse a single line from the JSONL file."""
        try:
            data = json.loads(line.strip())
            prediction = data.get('prediction', '')
            target_sequence = data.get('target_sequence', '')
            
            width = data.get('width', None)
            height = data.get('height', None)

            question_id = data.get("question_id", None)
            ann = self.ann[question_id]
            
            # Extract ground-truth bboxes
            gt_bboxes = self.get_gt_bbox(ann)
            gt_bboxes = [rescale_bbox(b, 224/width, 224/height) for b in gt_bboxes]


            # Extract reasoning and answer from prediction and target
            pred_reasoning_answer = self.extract_reasoning_and_answer(prediction)
            target_reasoning_answer = self.extract_reasoning_and_answer(target_sequence)
        
            # Compare sequences
            sequence_comparison = self.compare_sequences(
                pred_reasoning_answer['reasoning'],
                target_reasoning_answer['reasoning'],
                pred_reasoning_answer['answer'],
                target_reasoning_answer['answer'],
                gt_bboxes
            )
            
            result = {
                'question_id': data.get('question_id'),
                'image_file': data.get('image_file'),
                'width': width,
                'height': height,
                'prediction_raw': prediction,
                'target_raw': target_sequence,
                'comparison': sequence_comparison
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
    
    def calculate_dataset_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics across the entire dataset."""
        if not results:
            return {}
        
        metrics = {}
        
        # Answer accuracy
        answer_matches = [r['comparison']['answer_match'] for r in results]
        metrics['answer_accuracy'] = sum(answer_matches) / len(answer_matches)
        
        # Task sequence accuracy
        task_sequence_matches = [r['comparison']['task_comparison']['task_sequence_match'] for r in results]
        metrics['task_sequence_accuracy'] = sum(task_sequence_matches) / len(task_sequence_matches)
        
        # Average task order score
        task_order_scores = [r['comparison']['task_comparison']['task_order_score'] for r in results]
        metrics['avg_task_order_score'] = sum(task_order_scores) / len(task_order_scores)

        
        # Average Jaccard score
        # bbox_scores = [r['comparison']['task_comparison']['bbox_comparisons'][0] for r in results]
        # bbox_jaccard_scores = [s['jaccard'] for s in bbox_scores]
        # metrics['bbox_jaccard_scores'] = sum(bbox_jaccard_scores) / len(bbox_jaccard_scores)

        # # Average bbox IoU@0.5
        # bbox_iou_scores = [r['comparison']['bbox_iou_metrics']['avg_iou@0.5'] for r in results]
        # metrics['avg_bbox_iou@0.5'] = sum(bbox_iou_scores) / len(bbox_iou_scores)
        
        # Average raw IoU
        raw_iou_scores = [r['comparison']['bbox_iou_metrics']['avg_raw_iou'] for r in results]
        metrics['avg_raw_iou'] = sum(raw_iou_scores) / len(raw_iou_scores)
        
        return metrics


def main():
    parser = HeuristicbaseBboxParser(
        ann_file="/home/thodemon/workspace/composer-rebuild/dataset/heuristicbase/val/ann.jsonl",
        img_prefix="/home/thodemon/workspace/composer-rebuild/dataset/heuristicbase/val/images",
        grid_size=16
    )

    results = parser.parse("/home/thodemon/workspace/composer-rebuild/eval_heuristicbase/eval_heuristicbase/376261/checkpoint-final_0.jsonl")
    eval = parser.calculate_dataset_metrics(results)
    print(eval)
    
if __name__ == "__main__":
    main()