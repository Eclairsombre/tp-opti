## Pré-Requis

- Python
- uv

## Mise en place

Après avoir installé uv, pour synchroniser les dépendances :

```bash
uv sync
```

Marimo devrait être alors disponible via le terminal.

## Lancement

Pour lancer le projet avec Marimo :

Benchmark sur un algorithme :

```bash
marimo edit notebooks/benchmark.py --watch
```

Comparaison des deux algorithmes :

```bash
marimo edit notebooks/comparaison.py --watch
```

Visualisation des différentes solutions :

```bash
marimo edit notebooks/main.py --watch
```
