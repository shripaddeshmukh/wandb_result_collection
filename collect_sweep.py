from collections import defaultdict
import wandb
import pandas as pd
import argparse
from tqdm import tqdm


def main(project_id, sweep_id, output_config_file, output_results_file):
    api = wandb.Api()

    # Fetch the sweep
    try:
        sweep = api.sweep(path=f"{project_id}/{sweep_id}")
    except wandb.errors.CommError as e:
        print(f"Error fetching the sweep: {e}")
        return

    # Get runs from the sweep
    runs = list(sweep.runs)

    if not runs:
        print("No runs found in the sweep.")
        return

    # Assuming that all runs have the same structure for configs and summary
    hyperparameters = list(runs[-1].config.keys())
    results = list(runs[-1].history().keys())

    # Create a dataframe for the results
    df = pd.DataFrame(columns=hyperparameters + ["run_id", "name"])
    results_df = pd.DataFrame(columns=results + ["run_id", "name"])

    # Iterate over all runs
    for run in tqdm(runs):
        row = defaultdict(lambda: None)
        for hp in hyperparameters:
            if hp in run.config:
                row[hp] = run.config[hp]

        # for m in metrics:
        #     if m in run.summary:
        #         row[m] = run.summary[m]

        row["run_id"] = run.id
        row["name"] = run.name
        df = df._append(row, ignore_index=True)

        results_row = defaultdict(list)

        
        history_items = run.history(pandas=False)
        if history_items:
            for history_item in history_items:
                for r in results:
                    results_row[r].append(history_item[r])
        
        results_row["run_id"] = run.id
        results_row["name"] = run.name
        results_df = results_df._append(results_row, ignore_index=True)

    # Save the dataframe as a csv file
    try:
        df.to_csv(output_config_file, index=False)
        print("Saved config file to", output_config_file)
    except Exception as e:
        print(f"Error writing to {output_config_file}: {e}")

    try:
        results_df.to_csv(output_results_file, index=False)
        print("Saved results file to", output_results_file)
    except Exception as e:
        print(f"Error writing to {output_results_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data from wandb sweep runs and save them in csv files. This code will fetch the sweep runs from the project and save the hyperparameters and metrics in a csv file. It will also fetch the history of each run and save them in a separate csv file.")
    parser.add_argument("--project_id", type=str,
                        help="Project ID for the wandb project.")
    parser.add_argument("--sweep_id", type=str,
                        help="Sweep ID for the sweep within the project.")
    parser.add_argument("--output_config_file", type=str,
                        help="File path to save the sweep configurations.")
    parser.add_argument("--output_results_file", type=str,
                        help="File path to save the sweep results.")

    args = parser.parse_args()

    main(args.project_id, args.sweep_id,
         args.output_config_file, args.output_results_file)
