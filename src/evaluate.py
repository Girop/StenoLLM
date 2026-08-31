from dataclasses import dataclass
import argparse


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


def parse_log(logfile):
    with open(logfile, "r") as fp:
        lines = fp.readline()

    epochs = [
        eval(line[prefix:]) for line in lines if (prefix := line.find("Performance: "))
    ]

    final = None
    for line in lines:
        if (prefix := line.find("Final performance: ")):
            final = eval(line[prefix:])

    return epochs, final


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logs", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    log = parse_args().logs
    metrics, final = parse_log(log)

    for idx, item in enumerate(metrics):
        print(f"Epoch {idx}: {stats(item)}")
    print(f"Final: {stats(final)}")
