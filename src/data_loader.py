import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class EuroSATSubset(Dataset):
    """
    Clase personalizada para aplicar diferentes transformaciones (transforms) 
    a los subconjuntos sin que se filtren los aumentos de datos de entrenamiento 
    a los conjuntos de validación/prueba.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Obtenemos la imagen original (PIL) y su etiqueta
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]

        # Aplicamos la transformación correspondiente
        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(data_dir, batch_size=32, random_seed=42):
    """
    Carga el dataset EuroSAT, realiza una división estratificada 70/15/15 
    y devuelve los DataLoaders de PyTorch.
    """
    # 1. Cargar el dataset base sin transformaciones aún
    # (data_dir debería apuntar a 'data/raw/2750')
    base_dataset = datasets.ImageFolder(root=data_dir)
    targets = base_dataset.targets

    # 2. División Estratificada (Train: 70%, Temp: 30%)
    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.3,
        stratify=targets,
        random_state=random_seed
    )

    # 3. División del Temp en Validación (15%) y Prueba (15%)
    # Estratificamos usando las etiquetas correspondientes al temp_idx
    temp_targets = np.array(targets)[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_targets,
        random_state=random_seed
    )

    # 4. Definir Transformaciones (Data Augmentation)
    # Para el entrenamiento: Rotaciones, volteos y normalización
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),  # Convierte a tensor y escala de 0 a 1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Medias de ImageNet (estándar)
                             std=[0.229, 0.224, 0.225])
    ])

    # Para validación y prueba: SOLO convertir a tensor y normalizar (SIN aumentos)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. Instanciar nuestros Subsets con sus respectivas transformaciones
    train_dataset = EuroSATSubset(
        base_dataset, train_idx, transform=train_transform)
    val_dataset = EuroSATSubset(
        base_dataset, val_idx, transform=eval_transform)
    test_dataset = EuroSATSubset(
        base_dataset, test_idx, transform=eval_transform)

    # 6. Crear los DataLoaders (Generadores de lotes/batches)
    # shuffle=True solo en train para alimentar la red con aleatoriedad
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Extraer nombres de las clases para uso futuro
    class_names = base_dataset.classes

    print(f"Dataset cargado con éxito.")
    print(f"Clases encontradas: {len(class_names)}")
    print(f"Imágenes de Entrenamiento: {len(train_dataset)}")
    print(f"Imágenes de Validación: {len(val_dataset)}")
    print(f"Imágenes de Prueba: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# Bloque de prueba para ejecutar este script directamente desde la terminal
if __name__ == "__main__":
    # Ajusta la ruta a donde descomprimimos EuroSAT
    DATA_PATH = "../data/raw/2750"
    train_dl, val_dl, test_dl, classes = get_dataloaders(DATA_PATH)
