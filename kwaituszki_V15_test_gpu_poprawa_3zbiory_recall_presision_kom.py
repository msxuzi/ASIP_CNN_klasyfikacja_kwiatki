import os  # import modułu do operacji na systemie plików
import argparse  # import modułu do parsowania argumentów wiersza poleceń
import torch  # import biblioteki PyTorch
import torch.nn as nn  # import modułu sieci neuronowych z PyTorch
import torch.optim as optim  # import modułu optymalizacji z PyTorch
from torch.utils.data import DataLoader, random_split  # import narzędzi do ładowania danych i podziału zbiorów
from torchvision import datasets, transforms  # import narzędzi do przetwarzania obrazów
import matplotlib.pyplot as plt  # import biblioteki do tworzenia wykresów
import numpy as np  # import biblioteki do obliczeń numerycznych

# Nowe importy dla wyliczenia F1-score, Precision i Recall
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score  # import metryk z sklearn

# -------------------------------------------------------
# 1. Definicja stałych i ścieżek do danych
# -------------------------------------------------------
DATA_DIR = r"D:\uczelnia infa\II semestr\uczenie maszynowe\projekt drugi\projekt kwiatki\dane\flowers"  # ścieżka do danych
MODEL_PATH = "model_flowers.pth"  # nazwa pliku z zapisanym modelem

# -------------------------------------------------------
# 2. Transformacje danych (augmentacje, normalizacja itp.)
# -------------------------------------------------------
IMAGE_SIZE = 224  # rozmiar obrazu

train_transform = transforms.Compose([  # transformacje dla zbioru treningowego
    transforms.Resize((256, 256)),  # najpierw zmień rozmiar na 256x256
    transforms.RandomResizedCrop(IMAGE_SIZE),  # losowe przycięcie do 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # prawdopodobieństwo odbicia obrazu w poziomie
    transforms.RandomVerticalFlip(p=0.3),  # prawdopodobieństwo odbicia obrazu w pionie
    transforms.RandomRotation(degrees=15),  # losowy obrót o kąt do 15 stopni
    transforms.ColorJitter(brightness=0.2,  # zmiana jasności, kontrastu, nasycenia i barwy
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja obrazu
                         (0.229, 0.224, 0.225))
])

# Dla zbiorów walidacyjnego i testowego stosujemy prostsze transformacje (bez augmentacji)
val_transform = transforms.Compose([  # transformacje dla zbioru walidacyjnego i testowego
    transforms.Resize((256, 256)),  # zmiana rozmiaru na 256x256
    transforms.CenterCrop(IMAGE_SIZE),  # przycięcie do 224x224
    transforms.ToTensor(),  # konwersja obrazu do tensora
    transforms.Normalize((0.485, 0.456, 0.406),  # normalizacja obrazu
                         (0.229, 0.224, 0.225))
])

# -------------------------------------------------------
# 3. Stworzenie datasetu ImageFolder oraz podział na train, valid i test
# -------------------------------------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)  # załadowanie pełnego zbioru danych

# Ustal proporcje podziału (np. 70% trening, 15% walidacja, 15% test)
train_size = int(0.7 * len(full_dataset))  # obliczenie rozmiaru zbioru treningowego
val_size = int(0.15 * len(full_dataset))  # obliczenie rozmiaru zbioru walidacyjnego
test_size = len(full_dataset) - train_size - val_size  # obliczenie rozmiaru zbioru testowego

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])  # podział zbioru danych

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
class SimpleCNN(nn.Module):  # definicja klasy sieci neuronowej
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
            nn.Linear(64 * 4 * 4, 512),  # warstwa w pełni połączona
            nn.ReLU(),  # funkcja aktywacji ReLU
            nn.Dropout(p=0.25),  # warstwa dropout
            nn.Linear(512, num_classes)  # warstwa w pełni połączona
        )

    def forward(self, x):  # definicja funkcji forward
        x = self.features(x)  # przejście przez warstwy konwolucyjne
        x = x.view(x.size(0), -1)  # przekształcenie tensoru
        x = self.classifier(x)  # przejście przez warstwy klasyfikacyjne
        return x  # zwrócenie wyniku

# -------------------------------------------------------
# 6. Funkcje pomocnicze: trening i walidacja
# -------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):  # funkcja do trenowania jednej epoki
    model.train()  # ustawienie modelu w tryb treningowy
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania liczby wszystkich próbek

    for images, labels in dataloader:  # iteracja po batchach
        images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie
        optimizer.zero_grad()  # wyzerowanie gradientów

        outputs = model(images)  # przejście przez model
        loss = criterion(outputs, labels)  # obliczenie straty
        loss.backward()  # propagacja wsteczna
        optimizer.step()  # aktualizacja wag

        running_loss += loss.item() * images.size(0)  # aktualizacja straty
        _, predicted = torch.max(outputs.data, 1)  # predykcja
        total += labels.size(0)  # aktualizacja liczby wszystkich próbek
        correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

def validate(model, dataloader, criterion, device):  # funkcja do walidacji modelu
    """
    Zwraca (loss, accuracy) - do uproszczonego monitorowania treningu,
    natomiast precyzja, recall, F1 i macierz konfuzji będą liczone w
    osobnej funkcji 'plot_confusion_matrix' (albo w 'evaluate_metrics').
    """
    model.eval()  # ustawienie modelu w tryb ewaluacji
    running_loss = 0.0  # inicjalizacja zmiennej do przechowywania straty
    correct = 0  # inicjalizacja zmiennej do przechowywania liczby poprawnych predykcji
    total = 0  # inicjalizacja zmiennej do przechowywania liczby wszystkich próbek

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie

            outputs = model(images)  # przejście przez model
            loss = criterion(outputs, labels)  # obliczenie straty

            running_loss += loss.item() * images.size(0)  # aktualizacja straty
            _, predicted = torch.max(outputs.data, 1)  # predykcja
            total += labels.size(0)  # aktualizacja liczby wszystkich próbek
            correct += (predicted == labels).sum().item()  # aktualizacja liczby poprawnych predykcji

    epoch_loss = running_loss / total  # obliczenie straty na epokę
    epoch_acc = 100.0 * correct / total  # obliczenie dokładności na epokę
    return epoch_loss, epoch_acc  # zwrócenie straty i dokładności

# -------------------------------------------------------
# 7. Funkcja do trenowania sieci
# -------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):  # funkcja do trenowania modelu
    criterion = nn.CrossEntropyLoss()  # definicja funkcji straty
    optimizer = optim.Adam(model.parameters(), lr=lr)  # definicja optymalizatora

    train_losses, val_losses = [], []  # inicjalizacja list do przechowywania strat
    train_accs, val_accs = [], []  # inicjalizacja list do przechowywania dokładności

    for epoch in range(epochs):  # iteracja po epokach
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)  # trenowanie jednej epoki
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu

        train_losses.append(train_loss)  # dodanie straty treningowej do listy
        val_losses.append(val_loss)  # dodanie straty walidacyjnej do listy
        train_accs.append(train_acc)  # dodanie dokładności treningowej do listy
        val_accs.append(val_acc)  # dodanie dokładności walidacyjnej do listy

        print(f"Epoch [{epoch+1}/{epochs}], "  # wyświetlenie wyników epoki
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs  # zwrócenie strat i dokładności

# -------------------------------------------------------
# 8. Funkcja do tworzenia confusion matrix i dodatkowych miar
# -------------------------------------------------------
def evaluate_metrics(model, dataloader, device, class_names, set_name="Validation"):  # funkcja do ewaluacji metryk
    """
    Wyświetla macierz konfuzji i oblicza F1-score, Precision, Recall
    (zarówno per klasa, jak i 'weighted').
    set_name - nazwa zbioru, np. 'Validation' lub 'Test' (tylko do printowania).
    """
    model.eval()  # ustawienie modelu w tryb ewaluacji
    y_true = []  # inicjalizacja listy do przechowywania prawdziwych etykiet
    y_pred = []  # inicjalizacja listy do przechowywania przewidywanych etykiet

    with torch.no_grad():  # wyłączenie obliczania gradientów
        for images, labels in dataloader:  # iteracja po batchach
            images, labels = images.to(device), labels.to(device)  # przeniesienie danych na urządzenie
            outputs = model(images)  # przejście przez model
            _, predicted = torch.max(outputs, 1)  # predykcja
            y_true.extend(labels.cpu().numpy())  # dodanie prawdziwych etykiet do listy
            y_pred.extend(predicted.cpu().numpy())  # dodanie przewidywanych etykiet do listy

    # Macierz konfuzji
    cm = confusion_matrix(y_true, y_pred)  # obliczenie macierzy konfuzji

    plt.figure(figsize=(10, 8))  # ustawienie rozmiaru wykresu
    plt.pcolormesh(cm, cmap=plt.cm.Blues, edgecolors='none')  # rysowanie macierzy konfuzji
    plt.title(f"Macierz konfuzji - {set_name}")  # tytuł wykresu
    plt.colorbar(fraction=0.046, pad=0.04)  # dodanie paska kolorów
    tick_marks = np.arange(len(class_names)) + 0.5  # ustawienie znaczników osi
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)  # ustawienie etykiet osi X
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names)  # ustawienie etykiet osi Y

    # Dodanie wartości do komórek
    thresh = cm.max() / 2.  # próg dla koloru tekstu
    for i in range(cm.shape[0]):  # iteracja po wierszach
        for j in range(cm.shape[1]):  # iteracja po kolumnach
            plt.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),  # dodanie tekstu do komórki
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Rzeczywista klasa')  # etykieta osi Y
    plt.xlabel('Predykcja')  # etykieta osi X
    plt.tight_layout()  # dopasowanie wykresu
    plt.show()  # wyświetlenie wykresu

    # --- F1, Precision, Recall (per class i weighted) ---
    f1_per_class = f1_score(y_true, y_pred, average=None)  # obliczenie F1-score dla każdej klasy
    f1_weighted  = f1_score(y_true, y_pred, average='weighted')  # obliczenie F1-score ważonego

    precision_per_class = precision_score(y_true, y_pred, average=None)  # obliczenie precyzji dla każdej klasy
    precision_weighted  = precision_score(y_true, y_pred, average='weighted')  # obliczenie precyzji ważonej

    recall_per_class = recall_score(y_true, y_pred, average=None)  # obliczenie recall dla każdej klasy
    recall_weighted  = recall_score(y_true, y_pred, average='weighted')  # obliczenie recall ważonego

    print(f"=== {set_name} Metrics ===")  # wyświetlenie metryk
    print("F1-score per class:      ", f1_per_class)  # wyświetlenie F1-score dla każdej klasy
    print("F1-score (weighted):     ", f1_weighted)  # wyświetlenie F1-score ważonego

    print("Precision per class:     ", precision_per_class)  # wyświetlenie precyzji dla każdej klasy
    print("Precision (weighted):    ", precision_weighted)  # wyświetlenie precyzji ważonej

    print("Recall per class:        ", recall_per_class)  # wyświetlenie recall dla każdej klasy
    print("Recall (weighted):       ", recall_weighted)  # wyświetlenie recall ważonego
    print("======================================\n")  # zakończenie wyświetlania metryk

# -------------------------------------------------------
# 9. Główna część programu
# -------------------------------------------------------
def main():  # główna funkcja programu
    parser = argparse.ArgumentParser(description="Projekt: rozpoznawanie kwiatów.")  # parser argumentów wiersza poleceń
    parser.add_argument('--mode', type=str,  # argument trybu uruchomienia
                        help="Tryb uruchomienia: train lub eval.")
    parser.add_argument('--epochs', type=int, default=10,  # argument liczby epok
                        help='Liczba epok do trenowania (domyślnie 15).')
    parser.add_argument('--lr', type=float, default=0.001,  # argument learning rate
                        help='Learning rate (domyślnie 0.001).')
    args = parser.parse_args()  # parsowanie argumentów

    # Jeśli tryb nie został podany jako argument, zapytaj użytkownika
    mode = args.mode  # pobranie trybu uruchomienia
    if mode is None:  # jeśli tryb nie został podany
        mode = input("Wybierz tryb: wpisz 'train' aby trenować lub 'eval' aby wczytać model: ").strip()  # zapytaj użytkownika

    epochs = args.epochs  # pobranie liczby epok
    lr = args.lr  # pobranie learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ustawienie urządzenia

    if device.type == 'cuda':  # jeśli używamy GPU
        print("Używam GPU")  # wyświetlenie informacji
        print(f"Nazwa GPU: {torch.cuda.get_device_name(device)}")  # wyświetlenie nazwy GPU
    else:  # jeśli używamy CPU
        print("Używam CPU")  # wyświetlenie informacji

    num_classes = 5  # liczba klas
    model = SimpleCNN(num_classes=num_classes).to(device)  # stworzenie modelu

    class_names = full_dataset.classes  # pobranie nazw klas
    print(f"Wykryte klasy: {class_names}")  # wyświetlenie nazw klas

    if mode == 'train':  # jeśli tryb to 'train'
        print("Tryb: TRENING")  # wyświetlenie informacji
        print(f"Trenuję model przez {epochs} epok, LR = {lr}...")  # wyświetlenie informacji

        train_losses, val_losses, train_accs, val_accs = train_model(  # trenowanie modelu
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )

        print("Zapisuję wytrenowany model do pliku:", MODEL_PATH)  # wyświetlenie informacji
        torch.save(model.state_dict(), MODEL_PATH)  # zapisanie modelu

        # Ustawienie stylu wykresów
        plt.style.use('ggplot')  # ustawienie stylu wykresów

        # Wykres krzywej strat
        plt.figure(figsize=(10, 6))  # ustawienie rozmiaru wykresu
        plt.plot(range(1, epochs + 1), train_losses, label='Strata Treningowa', marker='o')  # rysowanie krzywej strat treningowych
        plt.plot(range(1, epochs + 1), val_losses, label='Strata Walidacyjna', marker='o')  # rysowanie krzywej strat walidacyjnych
        plt.title("Krzywa Strat w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Strata")  # etykieta osi Y
        plt.legend()  # legenda
        plt.grid(True)  # siatka
        plt.tight_layout()  # dopasowanie wykresu
        plt.savefig("loss_curve.png")  # zapisanie wykresu
        plt.show()  # wyświetlenie wykresu

        # Wykres krzywej dokładności
        plt.figure(figsize=(10, 6))  # ustawienie rozmiaru wykresu
        plt.plot(range(1, epochs + 1), train_accs, label='Dokładność Treningowa', marker='o')  # rysowanie krzywej dokładności treningowej
        plt.plot(range(1, epochs + 1), val_accs, label='Dokładność Walidacyjna', marker='o')  # rysowanie krzywej dokładności walidacyjnej
        plt.title("Krzywa Dokładności w Funkcji Epok")  # tytuł wykresu
        plt.xlabel("Epoka")  # etykieta osi X
        plt.ylabel("Dokładność (%)")  # etykieta osi Y
        plt.legend()  # legenda
        plt.grid(True)  # siatka
        plt.tight_layout()  # dopasowanie wykresu
        plt.savefig("accuracy_curve.png")  # zapisanie wykresu
        plt.show()  # wyświetlenie wykresu

        # Macierz konfuzji i miary dla zbioru walidacyjnego (na końcu treningu)
        evaluate_metrics(model, val_loader, device, class_names, set_name="Validation (po treningu)")  # ewaluacja metryk

    elif mode == 'eval':  # jeśli tryb to 'eval'
        print("Tryb: EWALUACJA")  # wyświetlenie informacji
        if not os.path.exists(MODEL_PATH):  # jeśli plik z modelem nie istnieje
            print(f"Nie znaleziono pliku z modelem: {MODEL_PATH}")  # wyświetlenie informacji
            print("Najpierw przeprowadź trening (tryb train).")  # wyświetlenie informacji
            return  # zakończenie funkcji

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # wczytanie modelu

        criterion = nn.CrossEntropyLoss()  # definicja funkcji straty

        # --- Walidacja ---
        val_loss, val_acc = validate(model, val_loader, criterion, device)  # walidacja modelu
        print(f"Strata Walidacyjna: {val_loss:.4f}, Dokładność Walidacyjna: {val_acc:.2f}%")  # wyświetlenie wyników walidacji
        evaluate_metrics(model, val_loader, device, class_names, set_name="Validation")  # ewaluacja metryk

        # --- Test ---
        test_loss, test_acc = validate(model, test_loader, criterion, device)  # testowanie modelu
        print(f"Strata Testowa: {test_loss:.4f}, Dokładność Testowa: {test_acc:.2f}%")  # wyświetlenie wyników testu
        evaluate_metrics(model, test_loader, device, class_names, set_name="Test")  # ewaluacja metryk

    else:  # jeśli tryb jest nieznany
        print(f"Nieznany tryb: {mode}. Użyj 'train' lub 'eval'.")  # wyświetlenie informacji

if __name__ == "__main__":  # jeśli skrypt jest uruchamiany bezpośrednio
    main()  # wywołanie głównej funkcji
