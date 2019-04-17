import torch
import torch.nn.functional as F
import numpy as np

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # training module
    for batch_idx, sample_batched in enumerate(train_loader):

        # for every batch, extract data and label (16, 1)
        data, label = sample_batched
        waveform, spectrogram, features, fmstd = data  # (16, 2, 240000), (16, 2, 1025, 431), (16, 10, 431), (16, 1, 10)

        # Map the variables to the current device (CPU or GPU)
        waveform = waveform.to(device, dtype=torch.float)
        spectrogram = spectrogram.to(device, dtype=torch.float)
        features = features.to(device, dtype=torch.float)
        fmstd = fmstd.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        # set initial gradients to zero :
        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/9
        optimizer.zero_grad()

        # pass the data into the model
        output = model(
            x_audio=waveform,
            x_spectrum=spectrogram,
            x_features=features,
            x_fmstd=fmstd
        )

        # get the loss using the predictions and the label
        loss = F.nll_loss(output, label)

        # backpropagate the losses
        loss.backward()

        # update the model parameters :
        # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        optimizer.step()

        # Printing the results
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, data_type):
    # evaluate the model
    model.eval()

    # init test loss
    test_loss = 0
    correct = 0
    print('Testing..')

    # Use no gradient backpropagations (as we are just testing)
    with torch.no_grad():
        # for every testing batch
        for i_batch, sample_batched in enumerate(test_loader):
            # for every batch, extract data and label (16, 1)
            data, label = sample_batched
            # (16, 2, 120000), (16, 2, 1025, 431), (16, 10, 431), (16, 2, 10)
            waveform, spectrogram, features, fmstd = data

            # Map the variables to the current device (CPU or GPU)
            waveform = waveform.to(device, dtype=torch.float)
            spectrogram = spectrogram.to(device, dtype=torch.float)
            features = features.to(device, dtype=torch.float)
            fmstd = fmstd.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            # get the predictions
            output = model(
                x_audio=waveform,
                x_spectrum=spectrogram,
                x_features=features,
                x_fmstd=fmstd
            )

            # accumulate the batchwise loss
            test_loss += F.nll_loss(output, label, reduction='sum').item()

            # get the predictions
            pred = output.argmax(dim=1, keepdim=True)

            # accumulate the correct predictions
            correct += pred.eq(label.view_as(pred)).sum().item()
    # normalize the test loss with the number of test samples
    test_loss /= len(test_loader.dataset)

    accuracy_percentage = 100. * correct / len(test_loader.dataset)
    # print the results
    print('Model prediction on ' + data_type + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy_percentage))
    return test_loss, accuracy_percentage


def evaluate(args, model, device, evaluate_loader):
    # evaluate the model
    model.eval()

    print('Testing..')
    predictions = []
    indexes = []        # It will help to reconstruct the order
    # Use no gradient backpropagations (as we are just testing)
    with torch.no_grad():
        # for every testing batch
        for i_batch, sample_batched in enumerate(evaluate_loader):
            # for every batch, extract data and label (16, 1)
            data, idx = sample_batched
            # (16, 2, 120000), (16, 2, 1025, 431), (16, 10, 431), (16, 2, 10)
            waveform, spectrogram, features, fmstd = data

            # Map the variables to the current device (CPU or GPU)
            waveform = waveform.to(device, dtype=torch.float)
            spectrogram = spectrogram.to(device, dtype=torch.float)
            features = features.to(device, dtype=torch.float)
            fmstd = fmstd.to(device, dtype=torch.float)

            # get the predictions
            output = model(
                x_audio=waveform,
                x_spectrum=spectrogram,
                x_features=features,
                x_fmstd=fmstd
            )

            # get the predictions
            predictions.extend(np.reshape(output.argmax(dim=1, keepdim=True).data.numpy(), (-1)).tolist())
            indexes.extend(idx.data.numpy().tolist())




            if i_batch % args.log_interval == 0:
                print('Evaluation : [{}/{} ({:.0f}%)]'.format(
                    i_batch * len(data), len(evaluate_loader.dataset),
                           100. * i_batch / len(evaluate_loader)))

    return predictions, indexes