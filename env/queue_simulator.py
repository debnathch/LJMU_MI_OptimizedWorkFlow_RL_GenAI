import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RadiologyCase:
    case_id: int
    arrival_time: int
    is_urgent: bool
    wait_time: int = 0
    assigned_radiologist: Optional[int] = None
    processed: bool = False

class QueueSimulator:
    """
    Simulates image arrival and manages the radiology case queue.
    """
    def __init__(self, config: dict):
        self.config = config
        self.arrival_rate = config['environment']['arrival_rate']
        self.urgency_prob = config['environment']['urgency_prob']
        self.max_queue_size = config['environment']['max_queue_size']
        self.queue: List[RadiologyCase] = []
        self.case_counter = 0

    def step(self, current_time: int):
        """
        Simulate arrivals for the current step using Poisson distribution.
        """
        num_arrivals = np.random.poisson(self.arrival_rate)
        for _ in range(num_arrivals):
            if len(self.queue) < self.max_queue_size:
                is_urgent = np.random.random() < self.urgency_prob
                case = RadiologyCase(
                    case_id=self.case_counter,
                    arrival_time=current_time,
                    is_urgent=is_urgent
                )
                self.queue.append(case)
                self.case_counter += 1

        # Update wait times for all cases in queue
        for case in self.queue:
            if not case.processed:
                case.wait_time = current_time - case.arrival_time

    def get_state(self):
        """
        Returns the current state of the queue.
        """
        urgent_count = sum(1 for c in self.queue if c.is_urgent)
        avg_wait = np.mean([c.wait_time for c in self.queue]) if self.queue else 0.0
        return {
            "queue_length": len(self.queue),
            "urgent_cases_count": urgent_count,
            "average_wait_time": avg_wait,
            "cases": self.queue
        }

    def remove_case(self, index: int) -> Optional[RadiologyCase]:
        """
        Removes a case from the queue for processing.
        """
        if 0 <= index < len(self.queue):
            return self.queue.pop(index)
        return None

    def reset(self):
        self.queue = []
        self.case_counter = 0
