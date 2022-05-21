import threading
import numpy as np
import pandas as pd
from .settings import OBSERVATION_COLUMNS


class StateCollector:
    def __init__(self):
        self.state = []
        self.state_access_lock = threading.Semaphore()

    def append_new_state(self, *args) -> None:
        """
        adds new state to the collection of values
        @param args:
        @return:
        """
        self.state_access_lock.acquire()
        self.state.append(np.array(args))
        self.state_access_lock.release()

    def empty_observed_state(self) -> None:
        self.state_access_lock.acquire()
        self.state = []
        self.state_access_lock.release()

    def get_observed_state(self) -> pd.DataFrame:
        self.state_access_lock.acquire()
        data_np = np.copy(np.vstack(self.state))
        self.state_access_lock.release()

        data_df = pd.DataFrame(data_np, columns=OBSERVATION_COLUMNS)

        return data_df
