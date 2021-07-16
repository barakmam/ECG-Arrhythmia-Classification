from data.dataset import Dataset
import ecg_plot


class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]





def plot_ecg(ecg):
    ecg_plot.plot(ecg, sample_rate=100, title='ECG 12', columns=1)
    ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    plot_ecg(b.X_train[0].T)
