"""
Script kiểm tra lỗi GMBL dataset
Kiểm tra 10 loại lỗi phổ biến trong GMBL code được generate từ LLM
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


class GMBLValidator:
    """Validator cho GMBL (Geometry Meaning-Based Language) code"""

    def __init__(self):
        # Các hàm cần được define riêng, không được nest trong assert
        self.nested_funcs = [
            'foot', 'midp', 'inter-ll', 'inter-lc', 'inter-cc',
            'incenter', 'circumcenter', 'excenter', 'orthocenter'
        ]

    def check_unbalanced_parens(self, answer: str) -> bool:
        """Kiểm tra số lượng dấu ngoặc có cân bằng không"""
        return answer.count('(') != answer.count(')')

    def check_extra_braces(self, answer: str) -> bool:
        """Kiểm tra các ký tự thừa như }} hoặc ]]"""
        return '}}' in answer or ']]' in answer

    def check_same_point_connecting(self, answer: str) -> bool:
        """
        Kiểm tra connecting với cùng 1 điểm: (connecting A A)
        WRONG: (connecting B B) - cần 2 điểm khác nhau
        """
        connecting_matches = re.findall(r'\(connecting\s+(\w+)\s+(\w+)\)', answer)
        for p1, p2 in connecting_matches:
            if p1 == p2:
                return True
        return False

    def check_inter_ll_same_line(self, answer: str) -> bool:
        """
        Kiểm tra inter-ll với cùng đường thẳng
        WRONG: (inter-ll (connecting A B) (connecting B A))
        """
        pattern = r'\(inter-ll\s+\(connecting\s+(\w+)\s+(\w+)\)\s+\(connecting\s+(\w+)\s+(\w+)\)\)'
        inter_ll_matches = re.findall(pattern, answer)
        for a, b, c, d in inter_ll_matches:
            # Kiểm tra nếu là cùng đường thẳng (A,B) = (C,D) hoặc (A,B) = (D,C)
            if (a == c and b == d) or (a == d and b == c):
                return True
        return False

    def check_circle_in_on_seg(self, answer: str) -> bool:
        """
        Kiểm tra circle được dùng trong on-seg
        WRONG: (define O circle (excircle A B C))
               (assert (on-seg O B C))
        Lý do: O là CIRCLE, không phải POINT, không thể dùng trong on-seg
        """
        # Tìm tất cả các biến có type là circle
        defines = re.findall(r'\(define\s+(\w+)\s+circle', answer)
        circle_vars = set(defines)

        # Tìm tất cả các biến được dùng trong on-seg
        on_seg_matches = re.findall(r'\(on-seg\s+(\w+)', answer)

        # Kiểm tra nếu có circle var trong on-seg
        for var in on_seg_matches:
            if var in circle_vars:
                return True
        return False

    def check_nested_in_assert(self, answer: str) -> bool:
        """
        Kiểm tra nested function trong assert
        WRONG: (assert (on-seg (foot A (connecting B C)) B C))
        RIGHT: (define F point (foot A (connecting B C)))
               (assert (on-seg F B C))
        """
        # Lấy tất cả các dòng assert
        assert_lines = [line.strip() for line in answer.split('\n')
                       if line.strip().startswith('(assert')]

        # Kiểm tra từng hàm nested
        for line in assert_lines:
            for func in self.nested_funcs:
                if f'({func} ' in line:
                    return True
        return False

    def check_wrong_incircle_args(self, answer: str) -> bool:
        """
        Kiểm tra số lượng tham số của incircle/excircle/circumcircle
        WRONG: (incircle A A) - cần 3 điểm khác nhau
        WRONG: (excircle A B) - thiếu điểm thứ 3
        RIGHT: (incircle A B C) - 3 điểm
        """
        # Tìm tất cả incircle/excircle/circumcircle
        incircle_matches = re.findall(r'\((?:incircle|excircle|circumcircle)\s+[^)]+\)', answer)

        for match in incircle_matches:
            # Tách lấy tên hàm và tham số
            # Ví dụ: "(incircle A B C)" -> ["incircle", "A", "B", "C"]
            parts = match.replace('(', '').replace(')', '').split()
            # parts[0] là tên hàm, parts[1:] là các tham số
            # Cần đúng 3 tham số (3 điểm)
            if len(parts) != 4:  # 1 tên hàm + 3 điểm = 4 parts
                return True
        return False

    def check_missing_on_seg_arg(self, answer: str) -> bool:
        """
        Kiểm tra on-seg thiếu tham số
        WRONG: (on-seg B C) - thiếu điểm thứ nhất
        RIGHT: (on-seg M B C) - M nằm trên đoạn BC
        """
        on_seg_full = re.findall(r'\(on-seg\s+([^)]+)\)', answer)

        for args in on_seg_full:
            parts = args.split()
            # on-seg cần ít nhất 3 phần: point + 2 điểm tạo segment
            # Hoặc: point + (connecting X Y)
            if len(parts) < 2:
                return True
            # Nếu có 2 parts nhưng có dấu ( => thiếu point
            # Ví dụ: (on-seg (connecting B C)) thiếu điểm
            if len(parts) == 2 and '(' in args:
                return True
        return False

    def check_wrong_cong_args(self, answer: str) -> bool:
        """
        Kiểm tra tham số của cong (congruent)

        Cong có 2 dạng:
        1. So sánh 2 đoạn thẳng: (cong A B C D) nghĩa là |AB| = |CD|
           Cần đúng 4 điểm

        2. So sánh 2 segments với connecting:
           (cong (connecting A B) (connecting C D))
           Cần đúng 2 lần connecting

        WRONG: (cong A B C D E F) - 6 điểm, sai!
        WRONG: (cong (connecting A B) C D) - trộn lẫn 2 dạng
        """
        cong_matches = re.findall(r'\(cong\s+([^)]+)\)', answer)

        for args in cong_matches:
            parts = args.split()

            if 'connecting' in args:
                # Dạng với connecting: phải có đúng 2 lần connecting
                if args.count('connecting') != 2:
                    return True
            else:
                # Dạng 4 điểm: phải có đúng 4 điểm
                if len(parts) != 4:
                    return True
        return False

    def check_wrong_type(self, answer: str) -> bool:
        """
        Kiểm tra type sai cho các hàm

        Các hàm trả về CIRCLE:
        - incircle, excircle, circumcircle, diam

        Các hàm trả về POINT:
        - incenter, excenter, circumcenter, orthocenter, foot, midp, inter-ll, inter-lc

        WRONG: (define O point (excircle A B C)) - excircle trả về circle
        RIGHT: (define O circle (excircle A B C))

        WRONG: (define O circle (excenter A B C)) - excenter trả về point
        RIGHT: (define O point (excenter A B C))
        """
        wrong_patterns = [
            # Point nhưng hàm trả về circle
            (r'\(define\s+\w+\s+point\s+\(excircle', 'excircle returns circle not point'),
            (r'\(define\s+\w+\s+point\s+\(incircle', 'incircle returns circle not point'),
            (r'\(define\s+\w+\s+point\s+\(circumcircle', 'circumcircle returns circle not point'),
            (r'\(define\s+\w+\s+point\s+\(diam', 'diam returns circle not point'),

            # Circle nhưng hàm trả về point
            (r'\(define\s+\w+\s+circle\s+\(excenter', 'excenter returns point not circle'),
            (r'\(define\s+\w+\s+circle\s+\(incenter', 'incenter returns point not circle'),
            (r'\(define\s+\w+\s+circle\s+\(circumcenter', 'circumcenter returns point not circle'),
            (r'\(define\s+\w+\s+circle\s+\(orthocenter', 'orthocenter returns point not circle'),
        ]

        for pattern, _ in wrong_patterns:
            if re.search(pattern, answer):
                return True
        return False

    def validate_answer(self, idx: int, answer: str) -> List[str]:
        """
        Kiểm tra 1 câu trả lời GMBL

        Args:
            idx: Index của sample
            answer: GMBL code cần kiểm tra

        Returns:
            List các loại lỗi tìm thấy
        """
        errors = []

        if self.check_unbalanced_parens(answer):
            errors.append('unbalanced_parens')

        if self.check_extra_braces(answer):
            errors.append('extra_braces')

        if self.check_same_point_connecting(answer):
            errors.append('same_point_connecting')

        if self.check_inter_ll_same_line(answer):
            errors.append('inter_ll_same_line')

        if self.check_circle_in_on_seg(answer):
            errors.append('circle_in_on_seg')

        if self.check_nested_in_assert(answer):
            errors.append('nested_in_assert')

        if self.check_wrong_incircle_args(answer):
            errors.append('wrong_incircle_args')

        if self.check_missing_on_seg_arg(answer):
            errors.append('missing_on_seg_arg')

        if self.check_wrong_cong_args(answer):
            errors.append('wrong_cong_args')

        if self.check_wrong_type(answer):
            errors.append('wrong_type')

        return errors


def evaluate_dataset(dataset_path: str) -> Dict:
    """
    Đánh giá toàn bộ dataset

    Args:
        dataset_path: Đường dẫn đến file JSON dataset

    Returns:
        Dict chứa kết quả đánh giá
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    validator = GMBLValidator()
    total = len(data)
    errors = []

    # Kiểm tra từng sample
    for idx, item in enumerate(data):
        answer = item.get('answer', '')
        errs = validator.validate_answer(idx, answer)
        if errs:
            errors.append({
                'idx': idx,
                'instruction': item.get('instruction', ''),
                'answer': answer,
                'errors': errs
            })

    # Tổng hợp lỗi theo loại
    error_types = {}
    for err_item in errors:
        for err_type in err_item['errors']:
            if err_type not in error_types:
                error_types[err_type] = []
            error_types[err_type].append(err_item['idx'])

    # Tính accuracy
    num_errors = len(errors)
    accuracy = ((total - num_errors) / total * 100) if total > 0 else 0

    return {
        'total': total,
        'correct': total - num_errors,
        'errors': num_errors,
        'accuracy': accuracy,
        'error_types': error_types,
        'error_details': errors
    }


def print_evaluation_summary(results: Dict, show_examples: bool = True, max_examples: int = 3):
    """
    In kết quả đánh giá

    Args:
        results: Kết quả từ evaluate_dataset
        show_examples: Có hiển thị ví dụ lỗi không
        max_examples: Số lượng ví dụ tối đa cho mỗi loại lỗi
    """
    print(f"{'='*60}")
    print(f"GMBL DATASET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Errors: {results['errors']}")
    print(f"Accuracy: {results['accuracy']:.1f}%")
    print()

    if results['error_types']:
        print(f"{'='*60}")
        print(f"ERROR BREAKDOWN")
        print(f"{'='*60}")

        # Sắp xếp theo số lượng lỗi giảm dần
        sorted_errors = sorted(results['error_types'].items(),
                              key=lambda x: len(x[1]),
                              reverse=True)

        for err_type, indices in sorted_errors:
            count = len(indices)
            percentage = (count / results['errors'] * 100) if results['errors'] > 0 else 0
            print(f"{err_type}: {count} ({percentage:.1f}% of errors)")
            print(f"  Samples: {indices[:10]}" + (' ...' if len(indices) > 10 else ''))

            if show_examples and count > 0:
                # Hiển thị ví dụ
                example_indices = indices[:max_examples]
                for i, idx in enumerate(example_indices, 1):
                    # Tìm error detail
                    detail = next((e for e in results['error_details'] if e['idx'] == idx), None)
                    if detail:
                        print(f"\n  Example {i} (sample {idx}):")
                        instruction = detail['instruction'][:80]
                        print(f"    Instruction: {instruction}...")
                        # Chỉ in 3 dòng đầu của answer
                        answer_lines = detail['answer'].split('\n')[:3]
                        answer_preview = '\n'.join(answer_lines)
                        print(f"    Answer:")
                        for line in answer_lines:
                            print(f"      {line}")
                        if len(detail['answer'].split('\n')) > 3:
                            print(f"      ...")
                print()
    else:
        print(f"\n{'='*60}")
        print(f"NO ERRORS FOUND - PERFECT DATASET!")
        print(f"{'='*60}")


if __name__ == '__main__':
    # Đường dẫn đến dataset
    dataset_path = 'dataset/generated_gmbl/train.json'

    print("Evaluating GMBL dataset...")
    print(f"Dataset: {dataset_path}")
    print()

    # Đánh giá
    results = evaluate_dataset(dataset_path)

    # In kết quả
    print_evaluation_summary(results, show_examples=True, max_examples=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")
