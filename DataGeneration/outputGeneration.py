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


def idx2label(idx):
    return labels[idx]


def label2idx(label):
    return labelsDict[label]


def return_predicted_labels(predictions_list):
    predicted_labels = []
    for i in range(len(predictions_list)):
        predicted_labels.append(labels[i])
    return predicted_labels


def create_csv(template_path, save_path, predictions_label, predictions, indexes):
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



