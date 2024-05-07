def calculate_accuracy(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()

    total_samples = len(lines)
    correct_predictions = 0

    for line in lines:
        filename, predicted_text = line.strip().split(': ')
        if filename == predicted_text:  # Jeśli przewidziana etykieta jest taka sama jak nazwa pliku
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

if __name__ == "__main__":
    results_file = "captcha_results.txt"
    accuracy = calculate_accuracy(results_file)
    print(f"Accuracy: {accuracy:.2f}%")
