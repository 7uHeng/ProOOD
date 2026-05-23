import torch
from torchmetrics.metric import Metric

class SCMetrics(Metric):
    def __init__(self, n_classes=2, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        
        self.n_classes = n_classes
        
        # self.add_state('tps', default=torch.zeros(
        #     self.n_classes), dist_reduce_fx='sum')
        # self.add_state('fps', default=torch.zeros(
        #     self.n_classes), dist_reduce_fx='sum')
        # self.add_state('fns', default=torch.zeros(
        #     self.n_classes), dist_reduce_fx='sum')
        
        self.add_state('completion_tp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fn', default=torch.zeros(1), dist_reduce_fx='sum')
    
    def compute_single(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        
        ret = (tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy())
        
        return ret
        
    def update(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
    
    def compute(self):
        precision = self.completion_tp / (self.completion_tp + self.completion_fp)
        recall = self.completion_tp / (self.completion_tp + self.completion_fn)
        iou = self.completion_tp / \
                (self.completion_tp + self.completion_fp + self.completion_fn)
        
        output = {
            "precision": precision,
            "recall": recall,
            "iou": iou.item(),
        }
        
        return output

    def get_score_completion(self, predict, target, nonempty=None):
        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1) # pts
        # predict = predict.view(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = torch.zeros_like(predict)
        b_true = torch.zeros_like(target)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = torch.sum((y_true == 1) & (y_pred == 1))
            fp = torch.sum((y_true != 1) & (y_pred == 1))
            fn = torch.sum((y_true == 1) & (y_pred != 1))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum
