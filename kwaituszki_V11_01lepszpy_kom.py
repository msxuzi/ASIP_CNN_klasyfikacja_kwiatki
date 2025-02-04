import os  # import modułu os do operacji na systemie plików
import argparse  # import modułu argparse do obsługi argumentów wiersza poleceń
import torch  # import biblioteki PyTorch do obliczeń tensorowych
import torch.nn as nn  # import modułu nn z PyTorch do tworzenia sieci neuronowych
import torch.optim as optim  # import modułu optim z PyTorch do optymalizacji
from torch.utils.data import DataLoader, random_split  # import DataLoader i random_split z PyTorch do ładowania danych
from torchvision import datasets, transforms  # import datasets i transforms z torchvision do przetwarzania obrazów
from torchvision.models import resnet18, ResNet18_Weights  # import modelu resnet18 i jego wag z torchvision
import matplotlib.pyplot as plt  # import matplotlib do tworzenia wykresów
import numpy as np  # import biblioteki numpy do obliczeń numerycznych
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # import confusion_matrix i ConfusionMatrixDisplay z sklearn do tworzenia macierzy niepewności

# -------------------------------------------------------
# 1. Definicja stałych i ścieżek do danych
# -------------------------------------------------------
DATA_DIR = r"D:\uczelnia infa\II semestr\uczenie maszynowe\projekt drugi\projekt kwiatki\dane\flowers"  # ścieżka do katalogu z danymi
MODEL_PATH = "model_flowers_resnet.pth"  # nazwa pliku z zapisanym modelem

# -------------------------------------------------------
# 2. Transformacje danych (augmentacje, normalizacja itp.)
# -------------------------------------------------------
# W przypadku transfer learningu w ResNet:
# - Często używa się rozdzielczości 224x224.
# - Poniżej bardziej rozbudowana augmentacja dla zbioru treningowego.
IMAGE_SIZE = 224  # rozmiar obrazu

train_transform = transforms.Compose([  # transformacje dla zbioru treningowego
    transforms.Resize((256, 256)),  # najpierw zmień rozmiar na 256x256
    transforms.RandomResizedCrop(IMAGE_SIZE),  # losowe przycięcie do 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # losowe odbicie poziome
    transforms.RandomVerticalFlip(p=0.3),  # losowe odbicie pionowe
    transforms.RandomRotation(degrees=15),  # losowy obrót o 15 stopni
    transforms.ColorJitter(brightness=0.2,  # losowa zmiana jasności
                           contrast=0.2,  # losowa zmiana kontrastu
                           saturation=0.2,  # losowa zmiana nasycenia
                           hue=0.1),  # losowa zmiana odcienia
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja obrazu
                         (0.229, 0.224, 0.225))  # statystyki ImageNet (typowe dla ResNet)
])

val_transform = transforms.Compose([  # transformacje dla zbioru walidacyjnego
    transforms.Resize((256, 256)),  # zmiana rozmiaru na 256x256
    transforms.CenterCrop(IMAGE_SIZE),  # przycięcie do 224x224
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja obrazu
                         (0.229, 0.224, 0.225))  # statystyki ImageNet (typowe dla ResNet)
])

# -------------------------------------------------------
# 3. Stworzenie datasetu ImageFolder oraz podział na train/val
# -------------------------------------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)  # stworzenie datasetu ImageFolder

train_size = int(0.8 * len(full_dataset))  # obliczenie rozmiaru zbioru treningowego (80% danych)
val_size = len(full_dataset) - train_size  # obliczenie rozmiaru zbioru walidacyjnego (20% danych)

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])  # podział datasetu na zbiór treningowy i walidacyjny

# Dla zbioru walidacyjnego podmieniamy transformacje na val_transform
val_dataset.dataset.transform = val_transform  # zmiana transformacji dla zbioru walidacyjnego

# -------------------------------------------------------
# 4. Stworzenie DataLoaderów
# -------------------------------------------------------
BATCH_SIZE = 32  # rozmiar batcha

train_loader = DataLoader(  # DataLoader dla zbioru treningowego
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # losowe mieszanie danych
    num_workers=4,  # liczba wątków do ładowania danych
    pin_memory=True if torch.cuda.is_available() else False  # przypinanie pamięci dla GPU
)

val_loader = DataLoader(  # DataLoader dla zbioru walidacyjnego
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # brak losowego mieszania danych
    num_workers=4,  # liczba wątków do ładowania danych
    pin_memory=True if torch.cuda.is_available() else False  # przypinanie pamięci dla GPU
)

# -------------------------------------------------------
# 5. Definicja modelu – transfer learning z ResNet18
# -------------------------------------------------------
class TransferLearningModel(nn.Module):  # definicja klasy modelu transfer learning
    def __init__(self, num_classes=5):  # konstruktor klasy
        super(TransferLearningModel, self).__init__()  # wywołanie konstruktora klasy bazowej
        # Ładujemy wstępnie wytrenowany model ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # załadowanie wstępnie wytrenowanego modelu ResNet18
        
        # ---------------------------------------------
        # (Opcjonalnie) zamrażamy wszystkie warstwy
        # aby trenować tylko głowicę (warstwę fc):
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # Jeżeli chcesz trenować wszystko w pełni,
        # możesz pominąć zamrażanie (lub zamrozić częściowo).
        # ---------------------------------------------
        
        # Podmieniamy końcową warstwę klasyfikacji:
        in_features = self.model.fc.in_features  # liczba wejść do warstwy fc
        self.model.fc = nn.Sequential(  # nowa warstwa fc
            nn.Linear(in_features, 512),  # warstwa liniowa
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.Dropout(0.5),  # warstwa dropout
            nn.Linear(512, num_classes)  # warstwa liniowa
        )

    def forward(self, x):  # metoda forward
        return self.model(x)  # wywołanie modelu

# -------------------------------------------------------
# 6. Funkcje pomocnicze: trening i walidacja
# -------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):  # funkcja do trenowania jednej epoki
    model.train()  # ustawienie modelu w tryb treningowy
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania liczby wszystkich próbek

    for images, labels in dataloader:  # iteracja po batchach
        images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)
        optimizer.zero_grad()  # wyzerowanie gradientów

        outputs = model(images)  # forward pass
        loss = criterion(outputs, labels)  # obliczenie straty
        loss.backward()  # backward pass
        optimizer.step()  # optymalizacja

        running_loss += loss.item() * images.size(0)  # aktualizacja straty
        _, predicted = torch.max(outputs.data, 1)  # predykcja
        total += labels.size(0)  # aktualizacja liczby wszystkich próbek
        correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

def validate(model, dataloader, criterion, device):  # funkcja do walidacji modelu
    model.eval()  # ustawienie modelu w tryb ewaluacji
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania liczby wszystkich próbek

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)

            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # obliczenie straty

            running_loss += loss.item() * images.size(0)  # aktualizacja straty
            _, predicted = torch.max(outputs.data, 1)  # predykcja
            total += labels.size(0)  # aktualizacja liczby wszystkich próbek
            correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

# -------------------------------------------------------
# 7. Funkcja do trenowania modelu
# -------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):  # funkcja do trenowania modelu
    criterion = nn.CrossEntropyLoss()  # funkcja straty
    # Można użyć np. AdamW, SGD z momentum, itp.
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optymalizator

    train_losses, val_losses = [], []  # listy do przechowywania strat
    train_accs, val_accs = [], []  # listy do przechowywania dokładności

    for epoch in range(epochs):  # iteracja po epokach
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)  # trenowanie jednej epoki
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu

        train_losses.append(train_loss)  # dodanie straty treningowej do listy
        val_losses.append(val_loss)  # dodanie straty walidacyjnej do listy
        train_accs.append(train_acc)  # dodanie dokładności treningowej do listy
        val_accs.append(val_acc)  # dodanie dokładności walidacyjnej do listy

        print(f"Epoch [{epoch+1}/{epochs}], "  # wypisanie wyników epoki
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs  # zwrócenie list strat i dokładności

# -------------------------------------------------------
# 8. Funkcja do tworzenia confusion matrix
# -------------------------------------------------------
def plot_confusion_matrix(model, dataloader, device, class_names):  # funkcja do tworzenia macierzy niepewności
    model.eval()  # ustawienie modelu w tryb ewaluacji
    y_true = []  # lista do przechowywania prawdziwych etykiet
    y_pred = []  # lista do przechowywania przewidywanych etykiet

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie (CPU/GPU)
            outputs = model(images)  # forward pass
            _, predicted = torch.max(outputs, 1)  # predykcja
            y_true.extend(labels.cpu().numpy())  # dodanie prawdziwych etykiet do listy
            y_pred.extend(predicted.cpu().numpy())  # dodanie przewidywanych etykiet do listy

    cm = confusion_matrix(y_true, y_pred)  # obliczenie macierzy niepewności
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)  # stworzenie obiektu do wyświetlania macierzy niepewności
    disp.plot(cmap=plt.cm.Blues)  # wyświetlenie macierzy niepewności
    plt.title("Macierz Niepewności")  # tytuł wykresu
    plt.show()  # pokazanie wykresu

# -------------------------------------------------------
# 9. Główna część programu
# -------------------------------------------------------
def main():  # główna funkcja programu
    parser = argparse.ArgumentParser(description="Projekt: rozpoznawanie kwiatów (transfer learning).")  # parser argumentów wiersza poleceń
    parser.add_argument('--mode', type=str,  # argument trybu uruchomienia
                        help="Tryb uruchomienia: train lub eval.")
    parser.add_argument('--epochs', type=int, default=20,  # argument liczby epok
                        help='Liczba epok do trenowania (domyślnie 10).')
    parser.add_argument('--lr', type=float, default=0.001,  # argument learning rate
                        help='Learning rate (domyślnie 0.001).')
    args = parser.parse_args()  # parsowanie argumentów

    # Jeśli tryb nie został podany jako argument, zapytaj użytkownika
    mode = args.mode  # pobranie trybu uruchomienia
    if mode is None:  # jeśli tryb nie został podany
        mode = input("Wybierz tryb: wpisz 'train' aby trenować lub 'eval' aby wczytać model: ").strip()  # zapytanie użytkownika o tryb

    epochs = args.epochs  # pobranie liczby epok
    lr = args.lr  # pobranie learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ustawienie urządzenia (CPU/GPU)
    if device.type == 'cuda':  # jeśli używane jest GPU
        print("Używam GPU")  # wypisanie informacji o używaniu GPU
    else:  # jeśli używane jest CPU
        print("Używam CPU")  # wypisanie informacji o używaniu CPU

    if device.type == 'cuda':  # jeśli używane jest GPU
        print(f"Nazwa GPU: {torch.cuda.get_device_name(device)}")  # wypisanie nazwy GPU
    # Wczytanie nazw klas
    class_names = full_dataset.classes  # pobranie nazw klas
    print(f"Wykryte klasy: {class_names}")  # wypisanie nazw klas

    # Zainicjalizowanie modelu transfer learning
    num_classes = len(class_names)  # liczba klas
    model = TransferLearningModel(num_classes=num_classes).to(device)  # stworzenie modelu transfer learning

    if mode == 'train':  # jeśli tryb to 'train'
        print("Tryb: TRENING")  # wypisanie trybu
        print(f"Trenuję model przez {epochs} epok, LR = {lr}...")  # wypisanie informacji o treningu

        train_losses, val_losses, train_accs, val_accs = train_model(  # trenowanie modelu
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )

        print("Zapisuję wytrenowany model do pliku:", MODEL_PATH)  # wypisanie informacji o zapisie modelu
        torch.save(model.state_dict(), MODEL_PATH)  # zapis modelu do pliku

        # Ustawienie stylu wykresów (opcjonalnie)
        plt.style.use('ggplot')  # ustawienie stylu wykresów

        # Wykres krzywej strat
        plt.figure(figsize=(10, 6))  # stworzenie nowego wykresu
        plt.plot(range(1, epochs + 1), train_losses, label='Strata Treningowa', marker='o')  # dodanie krzywej strat treningowych
        plt.plot(range(1, epochs + 1), val_losses, label='Strata Walidacyjna', marker='o')  # dodanie krzywej strat walidacyjnych
        plt.title("Krzywa Strat w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Strata")  # etykieta osi Y
        plt.legend()  # legenda
        plt.grid(True)  # siatka
        plt.tight_layout()  # dopasowanie layoutu
        plt.savefig("loss_curve.png")  # zapis wykresu do pliku
        plt.show()  # pokazanie wykresu

        # Wykres krzywej dokładności
        plt.figure(figsize=(10, 6))  # stworzenie nowego wykresu
        plt.plot(range(1, epochs + 1), train_accs, label='Dokładność Treningowa', marker='o')  # dodanie krzywej dokładności treningowej
        plt.plot(range(1, epochs + 1), val_accs, label='Dokładność Walidacyjna', marker='o')  # dodanie krzywej dokładności walidacyjnej
        plt.title("Krzywa Dokładności w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Dokładność (%)")  # etykieta osi Y
        plt.legend()  # legenda
        plt.grid(True)  # siatka
        plt.tight_layout()  # dopasowanie layoutu
        plt.savefig("accuracy_curve.png")  # zapis wykresu do pliku
        plt.show()  # pokazanie wykresu

        print("Macierz Niepewności (dla zbioru walidacyjnego):")  # wypisanie informacji o macierzy niepewności
        plot_confusion_matrix(model, val_loader, device, class_names)  # wyświetlenie macierzy niepewności

    elif mode == 'eval':  # jeśli tryb to 'eval'
        print("Tryb: EWALUACJA")  # wypisanie trybu
        if not os.path.exists(MODEL_PATH):  # jeśli plik z modelem nie istnieje
            print(f"Nie znaleziono pliku z modelem: {MODEL_PATH}")  # wypisanie informacji o braku pliku
            print("Najpierw przeprowadź trening (tryb train).")  # wypisanie informacji o konieczności treningu
            return  # zakończenie funkcji

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # wczytanie modelu

        criterion = nn.CrossEntropyLoss()  # funkcja straty
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu
        print(f"Strata Walidacyjna: {val_loss:.4f}, Dokładność Walidacyjna: {val_acc:.2f}%")  # wypisanie wyników walidacji

        plot_confusion_matrix(model, val_loader, device, class_names)  # wyświetlenie macierzy niepewności

    else:  # jeśli tryb jest nieznany
        print(f"Nieznany tryb: {mode}. Użyj 'train' lub 'eval'.")  # wypisanie informacji o nieznanym trybie


if __name__ == "__main__":  # jeśli plik jest uruchamiany bezpośrednio
    main()  # wywołanie głównej funkcji programu
