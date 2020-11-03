import torch
#import nbai

from torch.onnx.symbolic_helper import parse_args
@parse_args('v', 'v', 'f')
def symbolic_nmsrotated(g, dets, scores, iou_threshold):
    return g.op("nbai::nms_rotated", dets, scores, iou_f=iou_threshold)
from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('nbai::nms_rotated', symbolic_nmsrotated, 11)

torch.ops.load_library('./nbai.cpython-36m-x86_64-linux-gnu.so')

if __name__ == '__main__':
    boxes = torch.tensor([[50,50,70,100], [60,60,70,100], [10,10,50,50]])
    scores = torch.tensor([0.5,0.6,0.7])
    rotated_boxes = torch.zeros(3, 5)
    rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    rotated_boxes[0, 4] = 0
    rotated_boxes[1, 4] = 0
    print (rotated_boxes)

    class NmsRo(torch.nn.Module):
        def forward(self, boxes, scores):
            return torch.ops.nbai.nms_rotated(boxes, scores, 0.3)
    
    model = NmsRo()
    model.eval()

    print (model(rotated_boxes, scores))

    torch.onnx.export(model, (rotated_boxes, scores), 'nms_rotated.onnx', opset_version=11)
    # TODO: export to pt
    #ptmodel = torch.jit.trace(model, (rotated_boxes, scores))
    #ptmodel.save("nms_rotated.pt")
