# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import time
import logging
import mxnet as mx
import numpy as np


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class MXBoard(object):
    def __init__(self, tboardlog, frequent=50):
        assert tboardlog
        self.frequent = frequent
        self.tboardlog = tboardlog
        self.max_iter = 0
        self.iter = 0

    def __call__(self, param):
        """Callback to graph embs"""
        if param.epoch == 0:
            self.iter = param.nbatch
            if param.nbatch > self.max_iter:
                self.max_iter = param.nbatch
        else:
            self.iter = self.max_iter * param.epoch + param.nbatch

        if self.iter % self.frequent == 0:
            params = param[3]['self'].get_params()[0]  # pull params onto cpu (into the arg_params variable (param[3]['self'] is the module)
            for k,v in params.items():
                if k[:3] == 'fc_' or k[:3] == 'emb' or k[:3] == 'rpn' or k[:3] == 'bbo' or k[:3] == 'cls':
                    try:
                        self.tboardlog.add_histogram(tag=k,
                                                     values=v.asnumpy(),
                                                     bins=100,
                                                     global_step=self.iter)
                    except ValueError:
                        print("ValueError: range parameter must be finite: %s min: %f  max: %f" % (str(k), np.min(v.asnumpy()), np.max(v.asnumpy())))
                        print("You should really consider stopping training...")
            if param.eval_metric is not None:
                name, value = param.eval_metric.get()
                for n, v in zip(name, value):
                    self.tboardlog.add_scalar(tag=n, value=v, global_step=self.iter)


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('bbox_pred_weight_test')
        arg.pop('bbox_pred_bias_test')
    return _callback
