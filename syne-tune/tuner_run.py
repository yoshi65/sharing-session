#!/usr/bin/env python
# -*- coding: utf-8 -*-

from syne_tune.search_space import loguniform, uniform
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion


def run(scheduler):
    tuner = Tuner(
        backend=LocalBackend(entry_point="train_cifar10.py"),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=7200),
        n_workers=4,
    )

    tuner.run()


if __name__ == "__main__":
    max_epochs = 27
    config_space = {
        "epochs": max_epochs,
        "lr": loguniform(1e-5, 1e-1),
        "momentum": uniform(0.8, 1.0),
        "dropout_rate": loguniform(1e-5, 1.0),
    }

    scheduler = HyperbandScheduler(
        config_space,
        max_t=max_epochs,
        resource_attr='epoch',
        searcher='random',
        metric="val_acc",
        mode="max",
    )
    run(scheduler)

    scheduler = FIFOScheduler(
        config_space,
        searcher='random',
        metric="val_acc",
        mode="max",
    )
    run(scheduler)

    scheduler = HyperbandScheduler(
        config_space,
        max_t=max_epochs,
        resource_attr='epoch',
        searcher='bayesopt',
        metric="val_acc",
        mode="max",
    )
    run(scheduler)
