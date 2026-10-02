import json
import re
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
from composer.data.datasets.utils import rescale_bbox, reconstruct_patch_info_from_ground_truth, convert_patch_indices_to_bbox_xywh


class ComposerGCoTParser:
    def __init__(self,
                 results_file: str
                 ) -> None:
        
        self.results_file =  results_file     

    def extract_answer(self, 
                       text: str
                       ) -> str:
        
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        return answer_matches[0].strip() if answer_matches else ''
  
    def compare_answers(self, 
                          pred_answer: str, 
                          target_answer: str
                          ) -> bool:
        
        return pred_answer.strip().lower() == target_answer.strip().lower()
    
    
    def parse_jsonl_line(self, line: str) -> Dict[str, Any]:
        """Parse a single line from the JSONL file."""
        try:
            data = json.loads(line.strip())
            prediction = data.get('prediction', '')
            target_sequence = data.get('target_sequence', '')
           
            pred_answer = self.extract_answer(prediction)
            target_answer = target_sequence.strip()

            answer_match = self.compare_answers(pred_answer, target_answer)
            
            result = {
                'question_id': data.get('question_id'),
                'image_file': data.get('image_file'),
                'prediction': prediction,
                'target_sequence': target_sequence,
                'answer_match': answer_match
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
        except Exception as e:
            print(f"Error processing line: {e}")
            return None
    
    def parse(self) -> List[Dict[str, Any]]:
        results = []
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():  # Skip empty lines
                        parsed_line = self.parse_jsonl_line(line)
                        if parsed_line:
                            results.append(parsed_line)
                        else:
                            print(f"Failed to parse line {line_num}")
            
            return results
            
        except FileNotFoundError:
            print(f"File {self.results_file} not found")
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
        answer_matches = [r['answer_match'] for r in results]
        metrics['answer_accuracy'] = sum(answer_matches) / len(answer_matches)
        
        return metrics


def main():
    parser = ComposerGCoTParser(results_file="/home/thodemon/workspace/composer-rebuild/eval_heuristicbase/eval_heuristicbase/376440/checkpoint-final_0.jsonl")

    results = parser.parse()
    eval = parser.calculate_dataset_metrics(results)
    print(eval)
    
if __name__ == "__main__":
    main()