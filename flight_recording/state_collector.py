import threading
import numpy as np


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

    def get_observed_state(self) -> np.array:
        self.state_access_lock.acquire()
        data = np.copy(np.vstack(self.state))
        self.state_access_lock.release()

        return data
