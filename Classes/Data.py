
from torch.utils.data import Dataset

class IonDataset(Dataset):
    def __init__(self, file_train, file_test, file_val):
        self.X_train, self.y_train = self.read_file(file_train)
        self.X_test, self.y_test = self.read_file(file_test)
        self.X_val, self.y_val = self.read_file(file_val)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def read_file(self, file_path):
        x_data = []
        y_data = []

        with open(file_path) as f:
            next(f)
            data = f.readlines()
            for d in data:
                d = d.split(',')
                x_data.append(d[0])
                y_data.append(int(d[1].strip('\n')))

        return x_data, y_data


