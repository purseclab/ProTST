import pandas as pd

def calculate_metrics(correct_predictions, total_predictions, false_positives, false_negatives, class_labels):
    metrics = {}

    for key in total_predictions.keys():
        *elements, _ = key
        elements = tuple(elements)

        if elements in metrics:
            continue

        metrics[elements] = {}

        total_count = sum(total_predictions.get(elements + (class_label,), 0) for class_label in class_labels)

        for class_label in class_labels:  
            TP = correct_predictions.get(elements + (class_label,), 0)
            FP = false_positives.get(elements + (class_label,), 0)
            FN = false_negatives.get(elements + (class_label,), 0)
            total = total_predictions.get(elements + (class_label,), 0)
            
            Precision = TP / (TP + FP) if TP + FP != 0 else 0
            Recall = TP / (TP + FN) if TP + FN != 0 else 0
            F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0

            Accuracy = TP / total if total != 0 else 0
            
            metrics[elements][class_label] = {
                "Precision": Precision, 
                "Recall": Recall, 
                "F1 Score": F1, 
                "Accuracy": Accuracy, 
                "Weight": total / total_count if total_count != 0 else 0
            }

    for elements in metrics.keys():
        average = lambda metric: sum(metrics[elements][class_label][metric] for class_label in metrics[elements]) / len(metrics[elements])
        metrics[elements]["Average"] = {
            "Precision": average("Precision"),
            "Recall": average("Recall"),
            "F1 Score": average("F1 Score"),
            "Accuracy": average("Accuracy")
        }

    return metrics


def results_to_excel(metrics, result_loc, show_class=False):
    results = []
    for key, value in metrics.items():
        if show_class:
            for class_name, metric in value.items():
                result = {
                    'Setting' : key,
                    'Class': class_name,
                    'Precision': metric['Precision'],
                    'Recall': metric['Recall'],
                    'F1 Score': metric['F1 Score'],
                    'Weight': 0 if 'Weight' not in metric else metric['Weight']
                }
                results.append(result)
        
        else:
            metric = value['Average']
            result = {
                    'Setting' : key,
                    'Class': 'Average',
                    'Precision': metric['Precision'],
                    'Recall': metric['Recall'],
                    'F1 Score': metric['F1 Score'],
                    'Weight': 0 if 'Weight' not in metric else metric['Weight']
                }
            results.append(result)

    df = pd.DataFrame(results)
    df = df[['Setting', 'Class', 'Weight', 'Precision', 'Recall', 'F1 Score']]

    print('Optimization Average')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    df.to_excel(result_loc, index=False)

    class_avg = df.groupby('Class').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1 Score': 'mean'
    }).reset_index()

    print('Class Average')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(class_avg)

