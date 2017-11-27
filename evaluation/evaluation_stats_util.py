import sys
import re
import tagging_evaluation_util

def f1_metrics(total, total_gen, right):
    precision = right/float(total_gen) if total_gen != 0 else 0
    recall = right/float(total) if total != 0 else 0
    try:
        f1 = 2*recall*precision/(recall+precision)
    except ZeroDivisionError:
        f1 = 0

    return precision, recall, f1


class F1Stats(object):
    _PATTHERN_SPACE = re.compile(r'\s+', re.UNICODE)
    def __init__(self, is_fuzzy):
        self.is_fuzzy = is_fuzzy
        self.total = 0
        self.total_gen = 0
        self.right = 0

    def norm(self, text):
        if not isinstance(text, unicode): text = text.decode('utf-8')
        text = self._PATTHERN_SPACE.sub('', text)
        return text.lower()

    def is_correct(self, pred_ans, golden_answers):
        for golden_answer in golden_answers:
            if tagging_evaluation_util.is_right(pred_ans, golden_answer,
                    self.is_fuzzy):
                return True
        return False

    def update(self, golden_answers, pred_answers):
        # NOTE: self.norm() will remove space
        golden_answers = [self.norm(golden_answer) for golden_answer \
                in golden_answers]
        golden_answers = filter(None, golden_answers)

        self.total += 1
        self.total_gen += len(pred_answers)

        is_right = 0
        for pred_ans in pred_answers:
            pred_ans = self.norm(pred_ans)
            if self.is_correct(pred_ans, golden_answers):
                is_right = True
                break

        if is_right:
            self.right += 1
        
    def get_metrics_str(self):
        precision, recall, f1 = f1_metrics(self.total, self.total_gen, self.right) 
        return ("chunk_f1=%8.6f chunk_precision=%8.6f "
                "chunk_recall=%8.6f true_chunks=%d result_chunks=%d"
                " correct_chunks=%d") % \
                     (f1, precision, recall, self.total,
                             self.total_gen, self.right)
