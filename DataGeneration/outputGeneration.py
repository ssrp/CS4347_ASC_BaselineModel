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
