import torch
import os
import sys
import logging
import warnings
import json
import platform
from PIL import Image


from compat.config import Config
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from compat.path import mkdir_or_exist


warnings.filterwarnings('ignore')

class FAST:
    def __init__(self, config="config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py", checkpoint=None,
                 min_score=None, min_area=None, batch_size=1, worker=4, ema=False, cpu=False):
        self.config = config
        self.checkpoint = checkpoint
        self.min_score = min_score
        self.min_area = min_area
        self.batch_size = batch_size
        self.worker = worker
        self.ema = ema
        self.cpu = cpu


        os_name = platform.system()
        has_cuda = torch.cuda.is_available()
        #On OSX we can only use cpu
        if os_name == 'Darwin':  # macOS
            self.cpu = True
        elif not has_cuda:
            self.cpu = True
            print("Cuda is not installed on this machine, running on CPU.")

        self.cfg = Config.fromfile(self.config)
        if self.min_score is not None:
            self.cfg.test_cfg.min_score = self.min_score
        if self.min_area is not None:
            self.cfg.test_cfg.min_area = self.min_area

        self.cfg.batch_size = self.batch_size


    def has_text(self, image):
        return False

    def text(self, image):
        return ""


    def test(self, test_loader, model):
        rf = ResultFormat(self.cfg.data.test.type, self.cfg.test_cfg.result_path)

        results = dict()

        for idx, data in enumerate(test_loader):
            print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
            logging.info('Testing %d/%d\r' % (idx, len(test_loader)))
            # prepare input
            if not self.cpu:
                data['imgs'] = data['imgs'].cuda(non_blocking=True)
            data.update(dict(cfg=self.cfg))
            # forward
            with torch.no_grad():
                outputs = model(**data)

            # save result
            image_names = data['img_metas']['filename']
            for index, image_name in enumerate(image_names):
                rf.write_result(image_name, outputs['results'][index])
                results[image_name] = outputs['results'][index]

        if not self.cfg.report_speed:
            results = json.dumps(results)
            with open('outputs/output.json', 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, ensure_ascii=False)
                print("write json file success!")

    def main(self):
        # data loader
        data_loader = build_data_loader(self.cfg.data.test)
        test_loader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.worker,
            pin_memory=False
        )
        # model
        model = build_model(self.cfg.model)

        if not self.cpu:
            model = model.cuda()

        if self.checkpoint is not None:
            if os.path.isfile(self.checkpoint):
                print("Loading model and optimizer from checkpoint '{}'".format(self.checkpoint))
                logging.info("Loading model and optimizer from checkpoint '{}'".format(self.checkpoint))
                sys.stdout.flush()
                checkpoint = torch.load(self.checkpoint)

                if not self.ema:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint['ema']

                d = dict()
                for key, value in state_dict.items():
                    tmp = key.replace("module.", "")
                    d[tmp] = value
                model.load_state_dict(d)
            else:
                print("No checkpoint found at '{}'".format(self.checkpoint))
                raise

        model = rep_model_convert(model)

        # fuse conv and bn
        model = fuse_module(model)

        model.eval()
        self.test(test_loader, model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path', default="config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py")
    parser.add_argument('image',  type=str, default=None)
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    config_name = os.path.basename(args.config)

    tester = FAST(config=args.config, checkpoint=args.checkpoint,
                         min_score=args.min_score, min_area=args.min_area,
                         batch_size=args.batch_size, worker=args.worker, ema=args.ema, cpu=args.cpu)

    has_text = tester.has_text(Image.open(args.image))
    print(f"Has text: {has_text}")

    text = tester.text(Image.open(args.image))
    print(f"Text: {text}")
