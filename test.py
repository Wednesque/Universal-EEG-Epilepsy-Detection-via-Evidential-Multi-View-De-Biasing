import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
from data import MultiViewDataset
from models import EML
import pickle

BASE_DIR = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def validate(device, model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = num_samples = 0
        TP = TN = FP = FN = 0
        for batch in dataloader:
            x, y = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            view_e, fusion_e, loss, view_h = model(x)
            pred = fusion_e.cpu().argmax(dim=-1)
            correct += torch.sum(pred == y).item()
            num_samples += len(y)
            TP += torch.sum((pred == 1) & (pred == y)).item()
            TN += torch.sum((pred == 0) & (pred == y)).item()
            FP += torch.sum((pred == 1) & (pred != y)).item()
            FN += torch.sum((pred == 0) & (pred != y)).item()
    accuracy = correct / num_samples
    b_accuracy = (TN + TP) / (TP + TN + FP + FN)
    b_sensitivity = TP / (TP + FN)
    b_specificity = TN / (TN + FP)
    return {
        'accuracy': accuracy,
        'b_accuracy': b_accuracy, 'b_sensitivity': b_sensitivity, 'b_specificity': b_specificity
    }

def test_model(device, test_dataset, model_path):
    # Load dataset
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Load the model
    model = EML(sample_shapes=[s.shape for s in test_dataset[0]['x'].values()], num_classes=len(set(test_dataset.y)), device=device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    # Validate the model
    val = validate(device, model, test_loader)
    print('Test set results:', *[f'{k}:{v:.6f}' for k, v in val.items()])
    return val


def main():
    # Example usage
    #valid_path = BASE_DIR / 'dataset/eeg/data/domain_feature/data1_fold0_train.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = BASE_DIR / 'saved_models/0803_075_ModelFinalMerged_data_exclude3.pth'
    for i in range(1, 25):
        if i == 6:
            continue
        valid_path = BASE_DIR / f'dataset/eeg/data/domain_feature/data{i}_fold0_train.pkl'
        if valid_path.exists():
            print(f'Validating dataset {valid_path.name}...')
            test_dataset = MultiViewDataset(data_path=valid_path)
            test_results = test_model(device, test_dataset, model_path)
            print('\n')
        else:
            print(f'File {valid_path} does not exist.')
        valid_path = BASE_DIR / f'dataset/eeg/data/domain_feature/data{i}_fold0_valid.pkl'
        if valid_path.exists():
            print(f'Validating dataset {valid_path.name}...')
            test_dataset = MultiViewDataset(data_path=valid_path)
            test_results = test_model(device, test_dataset, model_path)
            print('\n\n\n')
        else:
            print(f'File {valid_path} does not exist.')




if __name__ == '__main__':
    main()
