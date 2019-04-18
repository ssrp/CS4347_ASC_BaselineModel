"""
    This file has the function used to predict and save the predictions during the evaluation task
"""


labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
          'street_pedestrian', 'street_traffic', 'tram']
labelsDict = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_sation': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9
}


def return_predicted_labels(predictions_list):
    """

    :param predictions_list: list of predictions (with the index in it)
    :return: list of predictions (with the labels in it)
    """
    predicted_labels = []
    for i in range(len(predictions_list)):
        predicted_labels.append(labels[predictions_list[i]])
    return predicted_labels


def create_csv(template_path, save_path, predictions_label, predictions, indexes):
    """

    :param template_path: the path to the .csv file of the evaluation_data (will be used as a template to create our .csv file)
    :param save_path: The path where we want to save our .csv file
    :param predictions_label: a list of prediction (with labels in it)
    :param predictions: a list of prediction (with indexes in it)
    :param indexes: The order of the labels in predictions_label and predictions (used to keep the order)

        Create the .csv file which will be used for the evaluation of the project
    """
    with open(template_path, 'r') as t:
        with open(save_path, 'w') as s:
            content = t.readlines()
            content = content[2:]
            s.write('filename,label,label_index\n')
            i = 0
            for path in content:
                if i % 2 == 0:
                    if i//2 == len(indexes):
                        break
                    path = path[:-1]
                    index = indexes.index(i//2)
                    line = '\n{0},{1},{2}\n'.format(path, predictions_label[index], predictions[index])
                    s.write(line)
                    i += 1
                else:
                    i += 1
