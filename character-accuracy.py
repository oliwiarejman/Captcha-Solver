def calculate_accuracy(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()

    total_samples = len(lines)
    total_characters = 0
    correct_characters = 0

    for line in lines:
        filename, predicted_text = line.strip().split(': ')
        filename_length = len(filename)
        predicted_length = len(predicted_text)
        
        total_characters += filename_length
        
        for i in range(min(filename_length, predicted_length)):
            if filename[i] == predicted_text[i]:
                correct_characters += 1
    
    accuracy = (correct_characters / total_characters) * 100
    
    return accuracy

if __name__ == "__main__":
    results_file = "captcha_results.txt"
    accuracy = calculate_accuracy(results_file)
    print(f"Character-level Accuracy: {accuracy:.2f}%")
