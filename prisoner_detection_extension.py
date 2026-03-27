import numpy as np
import pandas as pd


def detect_prisoner(expert_forecasts: pd.DataFrame):
    """
    Prisoner Detection extension.

    The idea is to identify experts who systematically react later
    to information compared to a leading expert (pioneer).
    """

    if expert_forecasts.shape[1] < 2:
        raise ValueError("Need at least two experts")

    pioneer = expert_forecasts.iloc[:, 0]

    delays = {}

    for expert in expert_forecasts.columns[1:]:
        corr = np.correlate(
            pioneer - pioneer.mean(),
            expert_forecasts[expert] - expert_forecasts[expert].mean(),
            mode="full"
        )

        lag = corr.argmax() - (len(pioneer) - 1)
        delays[expert] = lag

    return delays


if __name__ == "__main__":

    # small test example
    data = pd.DataFrame({
        "expert1": [2.4,2.2,1.9,1.88,1.80,1.75,1.73],
        "expert2": [2.9,2.8,2.6,2.5,2.4,2.3,2.2],
        "expert3": [2.8,2.7,2.6,2.4,2.1,1.9,1.8]
    })

    delays = detect_prisoner(data)

    print("Detected delays:")
    print(delays)