from dataclasses import dataclass
from math import e


@dataclass
class Metrics:
    successful_attacks: int = 0
    failed_attacks: int = 0
    false_triggers: int = 0
    no_action: int = 0

    def __add__(self, other):
        return Metrics(
                self.successful_attacks + other.successful_attacks,
                self.failed_attacks + other.failed_attacks,
                self.false_triggers + other.false_triggers,
                self.no_action + other.no_action
        )


len0 = [
    Metrics(successful_attacks=51, failed_attacks=11, false_triggers=51, no_action=15),
    Metrics(successful_attacks=61, failed_attacks=1, false_triggers=33, no_action=33),
    Metrics(successful_attacks=57, failed_attacks=5, false_triggers=22, no_action=44),
    Metrics(successful_attacks=58, failed_attacks=4, false_triggers=14, no_action=52),
    Metrics(successful_attacks=61, failed_attacks=1, false_triggers=24, no_action=42),
    Metrics(successful_attacks=61, failed_attacks=1, false_triggers=13, no_action=53),
    Metrics(successful_attacks=61, failed_attacks=1, false_triggers=18, no_action=48),
    Metrics(successful_attacks=59, failed_attacks=3, false_triggers=5, no_action=61),
    Metrics(successful_attacks=62, failed_attacks=0, false_triggers=15, no_action=51),
    Metrics(successful_attacks=62, failed_attacks=0, false_triggers=18, no_action=48),
    Metrics(successful_attacks=59, failed_attacks=3, false_triggers=7, no_action=59),
    Metrics(successful_attacks=60, failed_attacks=2, false_triggers=7, no_action=59),
    Metrics(successful_attacks=60, failed_attacks=2, false_triggers=11, no_action=55),
    Metrics(successful_attacks=62, failed_attacks=0, false_triggers=7, no_action=59),
]

m = Metrics(successful_attacks=56, failed_attacks=0, false_triggers=8, no_action=64)


def scenario_count(metrics: Metrics):
    return metrics.successful_attacks + metrics.failed_attacks + metrics.false_triggers + metrics.no_action

def asr(metrics: Metrics) -> float:
    """Attack success rate"""
    return metrics.successful_attacks / (metrics.successful_attacks + metrics.failed_attacks)


def ftr(metrics: Metrics) -> float:
    """False trigger rate"""
    return metrics.false_triggers / scenario_count(metrics)


def stats(metrics: Metrics) -> str:
    return f"Attack success rate = {asr(metrics):.2f}, False trigger rate = {ftr(metrics):.2f}"


for idx, item in enumerate(len0):
    print(f"Epoch {idx}: {stats(item)}")

print(f"Final: {stats(m)}")
