import torch.nn.utils
from utils import translate_sentence

def training(epochs, train_loader, model, optimizer, criterion, sentence, german, english, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        for idx, (source, target) in enumerate(train_loader):
            source, target = source.to(device), target.to(device)

            output = model(source, target)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        model.eval()
        translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)
        print(f"Translated example sentence: \n {translated_sentence}")
