import os
import csv

def check_folder_structure(root_path):
    """
    Check if the folder structure meets the requirements.
    :param root_path: The root folder path.
    :return: Whether the structure meets the requirements, and return an error message if not.
    """
    # Check if the variation folders exist
    for i in range(15):
        variation_path = os.path.join(root_path, str(i))
        if not os.path.isdir(variation_path):
            return False, f"Variation folder {i} does not exist"

        # Check the sub-task folders under each variation
        for sub_task in os.listdir(variation_path):
            sub_task_path = os.path.join(variation_path, sub_task)
            if os.path.isdir(sub_task_path):
                model_folder = None
                for item in os.listdir(sub_task_path):
                    if item.startswith("model_"):
                        model_folder = os.path.join(sub_task_path, item)
                        break
                if model_folder is None:
                    return False, f"No folder starting with 'model_' under sub-task {sub_task}"

                csv_path = os.path.join(model_folder, "eval_results.csv")
                if not os.path.isfile(csv_path):
                    return False, f"No 'eval_results.csv' file under the model folder of sub-task {sub_task}"

                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    if sum(1 for _ in reader) < 2:
                        return False, f"The 'eval_results.csv' file of sub-task {sub_task} has less than two lines, variation: {i}"

    return True, "The folder structure meets the requirements"

def collect_success_rates(root_path):
    """
    Collect the success rates of each task under each variation.
    :param root_path: The root folder path.
    :return: A dictionary containing the success rates of each task under each variation.
    """
    is_valid, message = check_folder_structure(root_path)
    if not is_valid:
        print(message)
        return None

    success_rates = {}
    for i in range(15):
        variation_path = os.path.join(root_path, str(i))
        for sub_task in os.listdir(variation_path):
            sub_task_path = os.path.join(variation_path, sub_task)
            if os.path.isdir(sub_task_path):
                for item in os.listdir(sub_task_path):
                    if item.startswith("model_"):
                        model_folder = os.path.join(sub_task_path, item)
                        csv_path = os.path.join(model_folder, "eval_results.csv")

                        with open(csv_path, 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            next(reader)  # Skip the header
                            row = next(reader)
                            task_name = row[0].rsplit('_', 1)[0]
                            success_rate = float(row[1])

                            if task_name not in success_rates:
                                success_rates[task_name] = [None] * 15
                            success_rates[task_name][i] = success_rate

    return success_rates

def save_to_csv(success_rates, output_path):
    """
    Save the success rates to a CSV file.
    :param success_rates: A dictionary containing the success rates of each task under each variation.
    :param output_path: The path of the output CSV file.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = ['Task'] + [f'Variation {i}' for i in range(15)]
        writer.writerow(headers)

        for task_name, rates in success_rates.items():
            row = [task_name] + [rate if rate is not None else '-' for rate in rates]
            writer.writerow(row)

        # Calculate and write the average success rate for each variation
        variation_sums = [0] * 15
        variation_counts = [0] * 15
        for rates in success_rates.values():
            for i, rate in enumerate(rates):
                if rate is not None:
                    variation_sums[i] += rate
                    variation_counts[i] += 1

        avg_rates = [variation_sums[i] / variation_counts[i] if variation_counts[i] > 0 else '-' for i in range(15)]
        writer.writerow(['Average'] + avg_rates)

if __name__ == "__main__":
    # Replace with the actual folder path
    folder_path="PATH_TO_RESULT_DIR"
    output_csv_path = "task_success_rates.csv"

    success_rates = collect_success_rates(folder_path)
    if success_rates:
        save_to_csv(success_rates, output_csv_path)
        print(f"Success rates have been saved to {output_csv_path}")
