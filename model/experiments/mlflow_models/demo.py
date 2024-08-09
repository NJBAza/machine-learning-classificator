import argparse
import os
import time

import mlflow


def eval(p1, p2):
    return p1**2 + p2**2


def main(input_1, input_2):
    mlflow.set_experiment("Demo_Experiment")
    #    with mlflow.start_run(run_name="Example Demo"):
    with mlflow.start_run():
        mlflow.set_tag("version", "1.0.0")
        mlflow.log_param("param1", input_1)
        mlflow.log_param("param2", input_2)
        metric = eval(p1=input_1, p2=input_2)
        mlflow.log_metric("Eval_Metric", metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifacts("dummy")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--param1", "-p1", type=int, default=5)
    args.add_argument("--param2", "-p2", type=int, default=10)
    parsed_args = args.parse_args()
    main(parsed_args.param1, parsed_args.param2)
