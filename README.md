# Spin System Simulation

Dieses Projekt erzeugt Monte-Carlo-Simulationen von zweidimensionalen
Spinsystemen und stellt Werkzeuge zur Auswertung der Simulationen bereit.
Dazu gehören Plot- und Mosaik-Visualisierungen, Frequenzanalysen sowie
Training und Evaluation neuronaler Netze.

## Projektstruktur

```text
.
├── configs/
│   ├── simulation.yaml.example       # Vorlage für die Parametererzeugung
│   ├── neural_network.yaml.example   # Vorlage für das Training der NNs
│   ├── simulation.yaml               # Lokale Einstellungen (nach setup)
│   ├── neural_network.yaml           # Lokale Hyperparameter (nach setup)
│   └── parameter_configs/            # Erzeugte Konfigurationen
├── src/spin_system/
│   ├── core/                         # Pfade, Konfigurationen und Dateiverwaltung
│   ├── generation/                   # Parameter, Simulationen, Fourier-Transformation und Plots
│   ├── neural_network/               # Training und Evaluation der NNs
│   ├── video/                        # Mosaik-Videoerstellung
│   └── cli.py                        # Kommandozeileninterface
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Installation

Voraussetzung ist eine unterstützte Python-Version (Python 3.10 oder neuer).
Im Projektverzeichnis erstellt der folgende Befehl eine virtuelle Umgebung
`.venv-BA`, installiert das Projekt samt Abhängigkeiten und richtet den
CLI-Befehl `lms` ein. Außerdem erstellt er fehlende lokale
Konfigurationsdateien aus den Vorlagen, ohne bestehende Dateien zu überschreiben:

```bash
make setup
```

Die Umgebung lässt sich anschließend direkt aktivieren:

```bash
source .venv-BA/bin/activate
lms --help
```

## Konfiguration und Speicherorte

Die Parameter für die Erzeugung von Simulationskonfigurationen stehen nach
`make setup` in `configs/simulation.yaml` (Vorlage:
[`configs/simulation.yaml.example`](configs/simulation.yaml.example)). Dort werden unter anderem
der Konfigurationsname, die Zufalls- oder Rastererzeugung, die Zahl der
Simulationen, die Gittergröße, Temperaturparameter, Hamilton-Terme und deren
Parameterbereiche festgelegt. Die Einstellungen fürs neuronale Netz stehen in
`configs/neural_network.yaml` (Vorlage:
[`configs/neural_network.yaml.example`](configs/neural_network.yaml.example)); sie können
global unter `default` und für einzelne Konfigurationen unter `configs`
definiert werden. Die lokalen Dateien können anschließend angepasst werden.

Die Projektpfade können über eine `.env`-Datei im Projektverzeichnis angepasst
werden. Eine Vorlage befindet sich in [`.env.example`](.env.example):

```dotenv
SPIN_SYSTEM_DATA_ROOT=/home/host/storage/data
SPIN_SYSTEM_MODEL_ROOT=/home/host/storage/models
SPIN_SYSTEM_OUTPUT_ROOT=/home/host/storage/output
```

`SPIN_SYSTEM_DATA_ROOT` legt den Speicherort der Daten fest,
`SPIN_SYSTEM_MODEL_ROOT` den der Modell-Dateien und
`SPIN_SYSTEM_OUTPUT_ROOT` den der erzeugten Ausgaben. Ohne eigene Werte
verwendet das Projekt die Verzeichnisse `data/`, `models/` und `outputs/` im
Projektverzeichnis. Zusätzlich kann `SPIN_SYSTEM_ROOT` das Projektverzeichnis
und `SPIN_SYSTEM_CONFIG_ROOT` das Konfigurationsverzeichnis festlegen. Die CLI
liest diese Einstellungen automatisch ein; vorhandene Umgebungsvariablen haben
Vorrang vor den Werten aus `.env`.

Simulationsdaten werden unter `<data-root>/<konfigurationsname>/` abgelegt.
Spin-Gitter liegen standardmäßig in `grids/`; Frequenzspektren liegen in
`magnitude/`. Modell- und Ausgabepfade werden jeweils nach Konfigurations- und
Modellnamen beziehungsweise Konfigurations- und Testsetnamen gruppiert.

## CLI-Befehle

Alle Projektfunktionen werden über `lms <Befehl>` aufgerufen. Die folgenden
Befehle sind unabhängig voneinander verfügbar; Optionen zu einem Befehl stehen
jeweils nach dessen Namen.

### `lms generator`

Erzeugt Parameterkonfigurationen anhand von `configs/simulation.yaml`. Bei
`generation_type: random` werden aktivierte Hamilton-Parameter zufällig aus
ihren konfigurierten Bereichen gewählt; bei `generation_type: grid` wird ein
Parametersatzraster erstellt. Die Ausgabe wird in
`configs/parameter_configs/` unter dem konfigurierten `config_name` gespeichert.



- `--seed <zahl>`: Optionaler Zufalls-Seed, um die zufällige Parametererzeugung
  reproduzierbar zu machen.

```bash
lms generator --seed 42
```

### `lms simulation`
#### Die Simulationen nutzen immer alle Kerne des Prozessors!

Führt die in einer Parameterkonfiguration beschriebenen Monte-Carlo-Läufe aus.
Ergebnisse werden als Spin-Gitter unter
`<data-root>/<config>/grids/` gespeichert.

- `-c`, `--config <name>`: Erforderlicher Konfigurationsname ohne
  `.yaml`-Erweiterung.
- `-f`, `--first_sim <name>`: Erste Simulation des auszuführenden Bereichs;
  Standard ist `sim_000001`.
- `-l`, `--last_sim <name>`: Letzte Simulation des auszuführenden Bereichs;
  Standard ist `sim_999999`. Der Bereich ist einschließlich Anfang und Ende.
- `--seed <zahl>`: Optionaler Seed für reproduzierbare, voneinander unabhängige
  Zufallsfolgen pro Lauf.

```bash
lms simulation -c <config> --seed 42
```

### `lms plot`

Erstellt Visualisierungen der Simulationsergebnisse.

- `-c`, `--config <name>`: Erforderlicher Konfigurationsname.
- `-t`, `--type <typ>`: Erforderlicher Plottyp: `imshow`, `quiver`, `mosaic`
  oder `mosaic_quiver`.
- `-f`, `--first_sim <name>` und `-l`, `--last_sim <name>`: Anfang und Ende
  des Simulationsbereichs; standardmäßig `sim_000001` bis `sim_999999`.
- `-s`, `--sim_name <name>`: Wählt genau eine Simulation aus.
- `--sim_names <name...>`: Wählt mehrere Simulationen anhand ihrer Namen und
  in der angegebenen Reihenfolge aus.
- `--sim_ids <zahl...>`: Wählt mehrere Simulationen anhand ihrer numerischen
  IDs aus, zum Beispiel `1 4` für `sim_000001` und `sim_000004`. Die Optionen
  `--sim_name`, `--sim_names` und `--sim_ids` schließen einander aus.
- `-d`, `--data <quelle...>`: Eine oder mehrere Datenquellen: `grids` oder
  `magnitude`. Standard ist `grids`.
- `--component <x|y|z...>`: Spin-Komponenten für den Plot. Für
  `--type imshow --data magnitude` ist mindestens eine Komponente erforderlich.
- `--dpi <zahl>`: Auflösung des Plots; Standard ist `300`.
- `--cmap <name>`: Matplotlib-Farbkarte; Standard ist `RdBu`.
- `--ratio <breite> <höhe>`: Seitenverhältnis; Standard ist `15 15`.
- `--show-info` / `--no-show-info`: Zeigt beziehungsweise verbirgt Titel und
  Hamilton-Parameter; standardmäßig werden sie gezeigt.
- `--show-axis-labels` / `--no-show-axis-labels`: Zeigt beziehungsweise
  verbirgt Achsenbeschriftungen und Tick-Werte; standardmäßig werden sie gezeigt.
- `--gap <pixel>`: Breite der Zwischenräume in Mosaiken; Standard ist `0`.
- `--gap-color <farbe>`: Farbe der Mosaik-Zwischenräume; Standard ist `white`.

```bash
lms plot -t mosaic -c <config> -f sim_000001 -l sim_001000
lms plot -t mosaic_quiver -c <config> --dpi 100 --ratio 10 5 --cmap viridis
lms plot -t imshow -c <config> --data magnitude --component z
```

### `lms video`

Erstellt ein MP4-Video im Format 3:2, das aus einem Mosaik der ausgewählten
Simulationen auf einen Ausschnitt mit sechs Simulationen zoomt. Ohne
`--output` wird das Video als
`<output-root>/videos/<config>/<config>_zoom.mp4` gespeichert.

- `-c`, `--config <name>`: Erforderlicher Konfigurationsname.
- `--source <quelle>`: Datenquelle; Standard ist `grids`.
- `-f`, `--first-sim <name>`: Optional erste einzubeziehende Simulation.
- `-l`, `--last-sim <name>`: Optional letzte einzubeziehende Simulation.
- `--duration <sekunden>`: Videolänge; Standard ist `8`.
- `--fps <zahl>`: Bilder pro Sekunde; Standard ist `30`.
- `--dpi <zahl>`: Auflösung; Standard ist `300`.
- `--cmap <name>`: Matplotlib-Farbkarte; Standard ist `berlin`.
- `--output <pfad>`: Zielpfad für die MP4-Datei.

```bash
lms video -c <config>
lms video -c <config> --first-sim sim_000001 --last-sim sim_001000 \
  --duration 10 --fps 30 --output ./simulation.mp4
```

### `lms freq`

Wandelt Spin-Gitter mit einer zweidimensionalen Fourier-Transformation in
zentrierte Log-Betragsspektren um und speichert sie als Datenquelle
`magnitude`. Die erzeugten Spektren können beispielsweise mit
`lms plot --data magnitude` dargestellt oder beim Netztraining verwendet werden.

- `-c`, `--config <name>`: Erforderlicher Konfigurationsname.
- `-s`, `--sim_name <name>`: Konvertiert eine einzelne Simulation.
- `-a`, `--all`: Konvertiert alle Simulationen der Konfiguration. `--sim_name`
  und `--all` schließen einander aus; eine der beiden Optionen ist erforderlich.

```bash
lms freq -c <config> --all
lms freq -c <config> --sim_name sim_000001
```

### `lms train`

Trainiert ein neuronales Netz auf den Simulationsergebnissen. Hyperparameter
werden aus `configs/neural_network.yaml` geladen und können dort
konfigurationsspezifisch angepasst werden. Checkpoint, Trainingsprotokoll und
Verlustkurven werden unter `<model-root>/<config>/<modellname>/` abgelegt.

- `-c`, `--config <name>`: Erforderlicher Name der Simulationskonfiguration.
- `-m`, `--model-save-name <name>`: Erforderlicher Modellname für die
  gespeicherten Trainingsartefakte.
- `-d`, `--data <quelle...>`: Eine oder mehrere Eingabequellen aus `grids` und
  `magnitude`; Standard ist `grids`. Bei mehreren Quellen werden die
  Eingabekanäle kombiniert.

```bash
lms train -c <config> -m <modellname>
lms train -c <config> -m <modellname> --data grids magnitude
```

### `lms evaluate`

Wertet ein trainiertes Modell auf einem Testdatensatz aus und erstellt
Kennzahlen, Vergleichs- und Fehlerplots sowie Mosaike für besonders gute und
schlechte Vorhersagen. Die Ergebnisse liegen unter
`<output-root>/evaluation/<config>/<modellname>/<testset>/`.

- `-c`, `--config <name>`: Erforderlicher Name der Simulationskonfiguration
  des Modells.
- `-m`, `--model-save-name <name>`: Erforderlicher Name des trainierten Modells.
- `-t`, `--test-set-name <name>`: Erforderlicher Name des Testsets.
- `-d`, `--data <quelle...>`: Eine oder mehrere Eingabequellen aus `grids` und
  `magnitude`; Standard ist `grids`. Die Auswahl muss mit den Datenquellen beim
  Training übereinstimmen.

```bash
lms evaluate -c <config> -m <modellname> -t <testset>
```

### `lms eval`

`eval` ist ein kürzerer Alias für `evaluate` und akzeptiert dieselben Optionen.

- `-c`, `--config <name>`: Erforderlicher Name der Simulationskonfiguration
  des Modells.
- `-m`, `--model-save-name <name>`: Erforderlicher Name des trainierten Modells.
- `-t`, `--test-set-name <name>`: Erforderlicher Name des Testsets.
- `-d`, `--data <quelle...>`: Eine oder mehrere Eingabequellen aus `grids` und
  `magnitude`; Standard ist `grids`. Die Auswahl muss mit den Datenquellen beim
  Training übereinstimmen.

```bash
lms eval -c <config> -m <modellname> -t <testset>
```

### Hilfe

`lms --help` zeigt die verfügbaren Befehle; `lms <Befehl> --help` zeigt die
Argumente des jeweiligen Befehls, zum Beispiel:

```bash
lms --help
lms plot --help
```

## Beispiel-Workflow

Ein typischer Ablauf ist, zunächst in `configs/simulation.yaml` die
Simulationsparameter festzulegen, danach Konfigurationen zu erzeugen und die
Simulationen auszuführen. Die erzeugten Spin-Gitter können anschließend geplottet
oder in den Frequenzbereich umgewandelt werden:

```bash
lms generator --seed 42
lms simulation -c <config> --seed 42
lms plot -t mosaic -c <config> -f sim_000001 -l sim_001000
lms freq -c <config> --all
lms plot -t imshow -c <config> --data magnitude --component z
```

Zum Training und zur Auswertung werden außerdem ein passendes Testset und eine
Neural-Network-Konfiguration benötigt:

```bash
lms train -c <config> -m <modellname>
lms evaluate -c <config> -m <modellname> -t <testset>
```
