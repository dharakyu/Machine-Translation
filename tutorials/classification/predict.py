from model import *
from data import *
import sys

rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

def simplePredict(input_line):
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        topv, topi = output.topk(1, 1, True)
        return all_categories[topi[0][0].item()]

def calcAccuracy(lines):
    total_examples = 0
    total_correct = 0
    for category in lines:
        for name in lines[category]:

            if simplePredict(name) == category: 
                total_correct += 1
            total_examples += 1

    print("n examples", total_examples)
    print("accuracy:", total_correct/total_examples)

print("Train:")
calcAccuracy(train_lines)
print("Test")
calcAccuracy(test_lines)
#predict(sys.argv[1])