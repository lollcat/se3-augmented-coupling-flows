import abc
from typing import Any, Dict, List, Mapping, Union, Optional

import pickle
import numpy as np
import pathlib
import wandb
import pandas as pd
import os

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    # copied from Acme: https://github.com/deepmind/acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


class ListLogger(Logger):
    """Manually save the data to the class in a dict. Currently only supports scalar history
    inputs."""

    def __init__(
        self,
        save: bool = False,
        save_path: str = "/tmp/logging_hist.pkl",
        save_period: int = 100,
    ):
        self.save = save
        self.save_path = save_path
        if save:
            if not pathlib.Path(self.save_path).parent.exists():
                pathlib.Path(self.save_path).parent.mkdir(exist_ok=True, parents=True)
        self.save_period = save_period  # how often to save the logging history
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.print_warning: bool = False
        self.iter = 0

    def write(self, data: LoggingData) -> None:
        for key, value in data.items():
            if key in self.history:
                try:
                    value = float(value)
                except:
                    pass
                self.history[key].append(value)
            else:  # add key to history for the first time
                if isinstance(value, np.ndarray):
                    assert np.size(value) == 1
                    value = float(value)
                else:
                    if isinstance(value, float) or isinstance(value, int):
                        pass
                    else:
                        if not self.print_warning:
                            print("non numeric history values being saved")
                            self.print_warning = True
                self.history[key] = [value]

        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            pickle.dump(
                self.history, open(self.save_path, "wb")
            )  # overwrite with latest version

    def close(self) -> None:
        if self.save:
            pickle.dump(self.history, open(self.save_path, "wb"))


class WandbLogger(Logger):
    def __init__(self, **kwargs: Any):
        self.run = wandb.init(**kwargs, reinit=True)
        self.iter: int = 0

    def write(self, data: Dict[str, Any]) -> None:
        self.run.log(data, step=self.iter, commit=False)
        self.iter += 1

    def close(self) -> None:
        self.run.finish()


class PandasLogger(Logger):
    """A pandas logger that writes all info into a single dataframe."""

    def __init__(
        self,
        save: bool = False,
        save_path: Optional[str] = None,
        save_period: int = 100,
    ):
        if save_path is None:
            self.save_path = "./logging_history.csv"
        self.save_path = os.path.join(save_path, "logging_history.csv")
        self.save = save
        self.save_period = save_period
        self.dataframe = pd.DataFrame()
        self.iter: int = 0
        if os.path.exists(self.save_path):
            self.dataframe = pd.read_csv(self.save_path, index_col=0)
            self.iter = int(self.dataframe.keys()[-1]) + 1

    def write(self, data: Dict[str, Any]) -> None:
        self.dataframe = self.dataframe.join(
            pd.Series(data, name=self.iter), how="outer"
        )
        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            if self.iter + 1 == self.save_period:  # First save
                if not pathlib.Path(self.save_path).parent.exists():
                    pathlib.Path(self.save_path).parent.mkdir(
                        exist_ok=True, parents=True
                    )
            self.dataframe.to_csv(
                open(self.save_path, "w")
            )  # overwrite with latest version

    def close(self) -> None:
        if self.save:
            self.dataframe.to_csv(
                open(self.save_path, "w")
            )  # overwrite with latest version
