import sys
import torch
import yaml
import logging
import argparse

from utils.general import check_suffix, check_yaml, check_file, check_dataset, colorstr, check_img_size, increment_path
from utils.torch_utils import intersect_dicts, select_device
from utils.downloads import attempt_download
from models.yolo import Model
from pathlib import Path
from utils.datasets import create_dataloader

import val  # for end-of-epoch mAP
from models.experimental import attempt_load


LOGGER = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--project', default=ROOT / 'runs/val_load', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":

    opt = parse_opt()
    
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False, mkdir=True))

    opt.data, opt.cfg, opt.hyp, opt.weights = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights)
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    cfg, hyp, weights, data, batch_size, workers, save_dir = opt.cfg, opt.hyp, opt.weights, opt.data, opt.batch_size, opt.workers, Path(opt.save_dir)

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # 
    device = select_device(opt.device, batch_size=opt.batch_size)
    data_dict = check_dataset(data)  # check if None
    nc = int(data_dict['nc'])  # number of classes
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # csd = ckpt['ema'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

# mAP val    
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple 

    val_path = data_dict['val']

    val_loader = create_dataloader(val_path, imgsz, batch_size, gs, single_cls=False,
                                    hyp=hyp, cache=opt.cache, rect=True, rank=-1,
                                    workers=workers, pad=0.5,
                                    prefix=colorstr('val: '))[0]

    _, _, _ = val.run(  data_dict,
                        batch_size=batch_size,
                        imgsz=imgsz,
                        model=model,        # attempt_load(weights, device).half(),
                        iou_thres=0.6,      # best pycocotools results at 0.7
                        single_cls=False,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=True,
                        plots=False
                        )