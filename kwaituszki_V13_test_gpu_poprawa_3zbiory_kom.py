import os                                              # importowanie modułu os do operacji na plikach i katalogach
import argparse                                        # importowanie modułu argparse do obsługi argumentów wiersza poleceń
import torch                                           # importowanie biblioteki PyTorch do obliczeń tensorowych
import torch.nn as nn                                  # importowanie modułu nn z PyTorch do budowy sieci neuronowych
import torch.optim as optim                            # importowanie modułu optim z PyTorch do optymalizacji
from torch.utils.data import DataLoader, random_split  # importowanie DataLoader i random_split do ładowania i podziału danych
from torchvision import datasets, transforms           # importowanie datasets i transforms z torchvision do przetwarzania obrazów
import matplotlib.pyplot as plt                        # importowanie matplotlib do tworzenia wykresów
import numpy as np                                     # importowanie numpy do obliczeń numerycznych
from sklearn.metrics import confusion_matrix           # importowanie confusion_matrix z sklearn do tworzenia macierzy niepewności

# -------------------------------------------------------
# 1. Definicja stałych i ścieżek do danych
# -------------------------------------------------------
DATA_DIR = r"D:\uczelnia infa\II semestr\uczenie maszynowe\projekt drugi\projekt kwiatki\dane\flowers"  # ścieżka do katalogu z danymi
MODEL_PATH = "model_flowers.pth"  # nazwa pliku z zapisanym modelem

# -------------------------------------------------------
# 2. Transformacje danych (augmentacje, normalizacja itp.)
# -------------------------------------------------------
IMAGE_SIZE = 224  # rozmiar obrazu wejściowego

train_transform = transforms.Compose([         # transformacje dla zbioru treningowego
    transforms.Resize((256, 256)),             # najpierw zmień rozmiar na 256x256
    transforms.RandomResizedCrop(IMAGE_SIZE),  # losowe przycięcie do 224x224
    transforms.RandomHorizontalFlip(p=0.5),    # losowe odbicie poziome
    transforms.RandomVerticalFlip(p=0.3),      # losowe odbicie pionowe
    transforms.RandomRotation(degrees=15),  # losowa rotacja o 15 stopni
    transforms.ColorJitter(brightness=0.2,  # losowa zmiana jasności
                           contrast=0.2,  # losowa zmiana kontrastu
                           saturation=0.2,  # losowa zmiana nasycenia
                           hue=0.1),  # losowa zmiana odcienia
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja z użyciem statystyk ImageNet
                         (0.229, 0.224, 0.225))
])

# Dla zbiorów walidacyjnego i testowego stosujemy prostszą transformację
val_transform = transforms.Compose([  # transformacje dla zbioru walidacyjnego i testowego
    transforms.Resize((256, 256)),  # zmiana rozmiaru na 256x256
    transforms.CenterCrop(IMAGE_SIZE),  # centralne przycięcie do 224x224
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja z użyciem statystyk ImageNet
                         (0.229, 0.224, 0.225))
])

# -------------------------------------------------------
# 3. Stworzenie datasetu ImageFolder oraz podział na train, valid i test
# -------------------------------------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)  # stworzenie datasetu z obrazami

# Ustal proporcje podziału (np. 70% trening, 15% walidacja, 15% test)
train_size = int(0.7 * len(full_dataset))  # obliczenie rozmiaru zbioru treningowego
val_size = int(0.15 * len(full_dataset))  # obliczenie rozmiaru zbioru walidacyjnego
test_size = len(full_dataset) - train_size - val_size  # obliczenie rozmiaru zbioru testowego

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])  # podział datasetu na zbiory

# Dla walidacji i testu stosujemy inne przekształcenia (bez augmentacji)
val_dataset.dataset.transform = val_transform  # ustawienie transformacji dla zbioru walidacyjnego
test_dataset.dataset.transform = val_transform  # ustawienie transformacji dla zbioru testowego

# -------------------------------------------------------
# 4. Stworzenie DataLoaderów
# -------------------------------------------------------
BATCH_SIZE = 32  # rozmiar batcha

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader dla zbioru treningowego
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # DataLoader dla zbioru walidacyjnego
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # DataLoader dla zbioru testowego

# -------------------------------------------------------
# 5. Definicja sieci CNN
# -------------------------------------------------------
class SimpleCNN(nn.Module):  # definicja klasy SimpleCNN dziedziczącej po nn.Module
    def __init__(self, num_classes=5):  # konstruktor klasy
        super(SimpleCNN, self).__init__()  # wywołanie konstruktora klasy bazowej
        self.features = nn.Sequential(  # definicja warstw konwolucyjnych
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # warstwa konwolucyjna
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # warstwa max pooling

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # warstwa konwolucyjna
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # warstwa max pooling

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # warstwa konwolucyjna
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.AdaptiveAvgPool2d((4, 4))  # warstwa adaptive average pooling
        )
        self.classifier = nn.Sequential(  # definicja warstw klasyfikacyjnych
            nn.Linear(64 * 4 * 4, 512),  # warstwa liniowa
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.Dropout(p=0.25),  # warstwa dropout
            nn.Linear(512, num_classes)  # warstwa liniowa
        )

    def forward(self, x):  # definicja funkcji forward
        x = self.features(x)  # przejście przez warstwy konwolucyjne
        x = x.view(x.size(0), -1)  # przekształcenie tensora
        x = self.classifier(x)  # przejście przez warstwy klasyfikacyjne
        return x  # zwrócenie wyniku

# -------------------------------------------------------
# 6. Funkcje pomocnicze: trening i walidacja
# -------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):  # funkcja do trenowania jednej epoki
    model.train()  # ustawienie modelu w tryb treningowy
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania całkowitej liczby próbek

    for images, labels in dataloader:  # iteracja po batchach
        images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)
        optimizer.zero_grad()  # wyzerowanie gradientów

        outputs = model(images)  # przejście przez model
        loss = criterion(outputs, labels)  # obliczenie straty
        loss.backward()  # propagacja wsteczna
        optimizer.step()  # aktualizacja wag

        running_loss += loss.item() * images.size(0)  # aktualizacja całkowitej straty
        _, predicted = torch.max(outputs.data, 1)  # predykcja etykiet
        total += labels.size(0)  # aktualizacja całkowitej liczby próbek
        correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie średniej straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

def validate(model, dataloader, criterion, device):  # funkcja do walidacji modelu
    model.eval()  # ustawienie modelu w tryb ewaluacji
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania całkowitej liczby próbek

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)

            outputs = model(images)  # przejście przez model
            loss = criterion(outputs, labels)  # obliczenie straty

            running_loss += loss.item() * images.size(0)  # aktualizacja całkowitej straty
            _, predicted = torch.max(outputs.data, 1)  # predykcja etykiet
            total += labels.size(0)  # aktualizacja całkowitej liczby próbek
            correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie średniej straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

# -------------------------------------------------------
# 7. Funkcja do trenowania sieci
# -------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cpu'):  # funkcja do trenowania modelu
    criterion = nn.CrossEntropyLoss()  # definicja funkcji straty
    optimizer = optim.Adam(model.parameters(), lr=lr)  # definicja optymalizatora

    train_losses, val_losses = [], []  # listy do przechowywania strat
    train_accs, val_accs = [], []  # listy do przechowywania dokładności

    for epoch in range(epochs):  # iteracja po epokach
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)  # trenowanie jednej epoki
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu

        train_losses.append(train_loss)  # dodanie straty treningowej do listy
        val_losses.append(val_loss)  # dodanie straty walidacyjnej do listy
        train_accs.append(train_acc)  # dodanie dokładności treningowej do listy
        val_accs.append(val_acc)  # dodanie dokładności walidacyjnej do listy

        print(f"Epoch [{epoch+1}/{epochs}], "  # wyświetlenie wyników dla epoki
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs  # zwrócenie list strat i dokładności

# -------------------------------------------------------
# 8. Funkcja do tworzenia confusion matrix
# -------------------------------------------------------
def plot_confusion_matrix(model, dataloader, device, class_names):  # funkcja do tworzenia macierzy niepewności
    model.eval()  # ustawienie modelu w tryb ewaluacji
    y_true = []  # lista do przechowywania rzeczywistych etykiet
    y_pred = []  # lista do przechowywania przewidywanych etykiet

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)
            outputs = model(images)  # przejście przez model
            _, predicted = torch.max(outputs, 1)  # predykcja etykiet
            y_true.extend(labels.cpu().numpy())  # dodanie rzeczywistych etykiet do listy
            y_pred.extend(predicted.cpu().numpy())  # dodanie przewidywanych etykiet do listy

    cm = confusion_matrix(y_true, y_pred)  # obliczenie macierzy niepewności

    plt.figure(figsize=(10, 8))  # ustawienie rozmiaru wykresu
    plt.pcolormesh(cm, cmap=plt.cm.Blues, edgecolors='none')  # użycie pcolormesh zamiast imshow
    plt.title("Macierz Niepewności")  # tytuł wykresu
    plt.colorbar(fraction=0.046, pad=0.04)  # dodanie paska kolorów
    tick_marks = np.arange(len(class_names)) + 0.5  # ustawienie znaczników osi
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)  # ustawienie etykiet osi X
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names)  # ustawienie etykiet osi Y

    thresh = cm.max() / 2.  # próg do zmiany koloru tekstu
    for i in range(cm.shape[0]):  # iteracja po wierszach macierzy
        for j in range(cm.shape[1]):  # iteracja po kolumnach macierzy
            plt.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),  # dodanie wartości do komórek
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Rzeczywistość')  # etykieta osi Y
    plt.xlabel('Predykcja')  # etykieta osi X
    plt.tight_layout()  # dopasowanie układu
    plt.show()  # wyświetlenie wykresu

# -------------------------------------------------------
# 9. Główna część programu
# -------------------------------------------------------
def main():  # główna funkcja programu
    parser = argparse.ArgumentParser(description="Projekt: rozpoznawanie kwiatów.")  # parser argumentów wiersza poleceń
    parser.add_argument('--mode', type=str,  # dodanie argumentu 'mode'
                        help="Tryb uruchomienia: train lub eval.")
    parser.add_argument('--epochs', type=int, default=15,  # dodanie argumentu 'epochs'
                        help='Liczba epok do trenowania (domyślnie 15).')
    parser.add_argument('--lr', type=float, default=0.001,  # dodanie argumentu 'lr'
                        help='Learning rate (domyślnie 0.001).')
    args = parser.parse_args()  # parsowanie argumentów

    mode = args.mode  # pobranie wartości argumentu 'mode'
    if mode is None:  # jeśli tryb nie został podany jako argument
        mode = input("Wybierz tryb: wpisz 'train' aby trenować lub 'eval' aby wczytać model: ").strip()  # zapytaj użytkownika

    epochs = args.epochs  # pobranie wartości argumentu 'epochs'
    lr = args.lr  # pobranie wartości argumentu 'lr'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ustawienie urządzenia (CPU/GPU)

    if device.type == 'cuda':  # jeśli używane jest GPU
        print("Używam GPU")  # wyświetlenie informacji o używaniu GPU
        print(f"Nazwa GPU: {torch.cuda.get_device_name(device)}")  # wyświetlenie nazwy GPU
    else:  # jeśli używane jest CPU
        print("Używam CPU")  # wyświetlenie informacji o używaniu CPU

    num_classes = 5  # liczba klas
    model = SimpleCNN(num_classes=num_classes).to(device)  # stworzenie modelu i przeniesienie na urządzenie

    class_names = full_dataset.classes  # pobranie nazw klas
    print(f"Wykryte klasy: {class_names}")  # wyświetlenie nazw klas

    if mode == 'train':  # jeśli tryb to 'train'
        print("Tryb: TRENING")  # wyświetlenie informacji o trybie treningowym
        print(f"Trenuję model przez {epochs} epok, LR = {lr}...")  # wyświetlenie informacji o liczbie epok i learning rate

        train_losses, val_losses, train_accs, val_accs = train_model(  # trenowanie modelu
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )

        print("Zapisuję wytrenowany model do pliku:", MODEL_PATH)  # wyświetlenie informacji o zapisywaniu modelu
        torch.save(model.state_dict(), MODEL_PATH)  # zapisanie modelu

        plt.style.use('ggplot')  # ustawienie stylu wykresów

        plt.figure(figsize=(10, 6))  # ustawienie rozmiaru wykresu
        plt.plot(range(1, epochs + 1), train_losses, label='Strata Treningowa', marker='o')  # wykres krzywej strat treningowych
        plt.plot(range(1, epochs + 1), val_losses, label='Strata Walidacyjna', marker='o')  # wykres krzywej strat walidacyjnych
        plt.title("Krzywa Strat w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Strata")  # etykieta osi Y
        plt.legend()  # dodanie legendy
        plt.grid(True)  # dodanie siatki
        plt.tight_layout()  # dopasowanie układu
        plt.savefig("loss_curve.png")  # zapisanie wykresu do pliku
        plt.show()  # wyświetlenie wykresu

        plt.figure(figsize=(10, 6))  # ustawienie rozmiaru wykresu
        plt.plot(range(1, epochs + 1), train_accs, label='Dokładność Treningowa', marker='o')  # wykres krzywej dokładności treningowych
        plt.plot(range(1, epochs + 1), val_accs, label='Dokładność Walidacyjna', marker='o')  # wykres krzywej dokładności walidacyjnych
        plt.title("Krzywa Dokładności w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Dokładność (%)")  # etykieta osi Y
        plt.legend()  # dodanie legendy
        plt.grid(True)  # dodanie siatki
        plt.tight_layout()  # dopasowanie układu
        plt.savefig("accuracy_curve.png")  # zapisanie wykresu do pliku
        plt.show()  # wyświetlenie wykresu

        print("Macierz niepewności (zbiór walidacyjny):")  # wyświetlenie informacji o macierzy niepewności
        plot_confusion_matrix(model, val_loader, device, class_names)  # wyświetlenie macierzy niepewności dla zbioru walidacyjnego

    elif mode == 'eval':  # jeśli tryb to 'eval'
        print("Tryb: EWALUACJA")  # wyświetlenie informacji o trybie ewaluacji
        if not os.path.exists(MODEL_PATH):  # jeśli plik z modelem nie istnieje
            print(f"Nie znaleziono pliku z modelem: {MODEL_PATH}")  # wyświetlenie informacji o braku pliku z modelem
            print("Najpierw przeprowadź trening (tryb train).")  # wyświetlenie informacji o konieczności przeprowadzenia treningu
            return  # zakończenie programu

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # wczytanie modelu

        criterion = nn.CrossEntropyLoss()  # definicja funkcji straty
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu
        print(f"Strata Walidacyjna: {val_loss:.4f}, Dokładność Walidacyjna: {val_acc:.2f}%")  # wyświetlenie wyników walidacji
        plot_confusion_matrix(model, val_loader, device, class_names)  # wyświetlenie macierzy niepewności dla zbioru walidacyjnego

        test_loss, test_acc = validate(model, test_loader, criterion, device)  # ocena na zbiorze testowym
        print(f"Strata Testowa: {test_loss:.4f}, Dokładność Testowa: {test_acc:.2f}%")  # wyświetlenie wyników testu
        plot_confusion_matrix(model, test_loader, device, class_names)  # wyświetlenie macierzy niepewności dla zbioru testowego

    else:  # jeśli tryb jest nieznany
        print(f"Nieznany tryb: {mode}. Użyj 'train' lub 'eval'.")  # wyświetlenie informacji o nieznanym trybie

if __name__ == "__main__":  # jeśli skrypt jest uruchamiany bezpośrednio
    main()  # wywołanie głównej funkcji programu
