import torch
import os
import sys
import logging
import warnings
import json

from compat.config import Config
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter

warnings.filterwarnings('ignore')

class FAST:
    def __init__(self, config="config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py", checkpoint=None, report_speed=False, print_model=False,
                 min_score=None, min_area=None, batch_size=1, worker=4, ema=False, cpu=False):
        self.config = config
        self.checkpoint = checkpoint
        self.report_speed = report_speed
        self.print_model = print_model
        self.min_score = min_score
        self.min_area = min_area
        self.batch_size = batch_size
        self.worker = worker
        self.ema = ema
        self.cpu = cpu

        self.cfg = Config.fromfile(self.config)
        for d in [self.cfg, self.cfg.data.test]:
            d.update(dict(
                report_speed=self.report_speed,
            ))
        if self.min_score is not None:
            self.cfg.test_cfg.min_score = self.min_score
        if self.min_area is not None:
            self.cfg.test_cfg.min_area = self.min_area

        self.cfg.batch_size = self.batch_size

    def report_speed(self, model, data, speed_meters, batch_size=1, times=10):
        for _ in range(times):
            total_time = 0
            outputs = model(**data)
            for key in outputs:
                if 'time' in key:
                    speed_meters[key].update(outputs[key] / batch_size)
                    total_time += outputs[key] / batch_size
            speed_meters['total_time'].update(total_time)
            for k, v in speed_meters.items():
                print('%s: %.4f' % (k, v.avg))
                logging.info('%s: %.4f' % (k, v.avg))
            print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))
            logging.info('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))

    def test(self, test_loader, model):
        rf = ResultFormat(self.cfg.data.test.type, self.cfg.test_cfg.result_path)

        if self.cfg.report_speed:
            speed_meters = dict(
                backbone_time=AverageMeter(1000 // self.batch_size),
                neck_time=AverageMeter(1000 // self.batch_size),
                det_head_time=AverageMeter(1000 // self.batch_size),
                post_time=AverageMeter(1000 // self.batch_size),
                total_time=AverageMeter(1000 // self.batch_size)
            )
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

            if self.cfg.report_speed:
                self.report_speed(model, data, speed_meters, self.cfg.batch_size)
                continue

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

    def model_structure(self, model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  ##如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

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

        if self.print_model:
            self.model_structure(model)

        model.eval()
        self.test(test_loader, model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path', default="config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py")
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report-speed', action='store_true')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    mmcv.mkdir_or_exist("./speed_test")
    config_name = os.path.basename(args.config)
    logging.basicConfig(filename=f'./speed_test/{config_name}.txt', level=logging.INFO)

    tester = FAST(config=args.config, checkpoint=args.checkpoint, report_speed=args.report_speed,
                         print_model=args.print_model, min_score=args.min_score, min_area=args.min_area,
                         batch_size=args.batch_size, worker=args.worker, ema=args.ema, cpu=args.cpu)
    tester.main()
