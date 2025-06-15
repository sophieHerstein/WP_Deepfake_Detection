# Wissenschaftliches Projekt - Deepfake Detection Modellvergleich

## Aufbau des Projekts

- `data/`: Enthält die Datensätze für Training und Testing.
    - `celebdf_ff/`: Trainings- und Testdaten für Testszenario 2 und 3
    - `celebdf_only/`: Trainings- und Testdaten für Testszenario 1
    - `images`: initiale Datensätze
- `gradcam/`: Enthält die Grad-CAM Visualisierungen der Modelle untergliedert nach Testszenario, Modell und Augmnetierungsart
- `logs/`: Enthält die Logdateien der Trainings- und Testläufe unterglieder nach Trainings-Szenario bzw. Test-Szenario und Augmentierungsart
- `plots/`: Enthält die Plots, die während des Trainings oder auch danach generiert wurden
- `results/`: Enthält die Ergebnisse der Testszenarien in Form von CSV-Dateien untergliedert nach Testszenario, Modell und Augmentierungsart
- `util/`: Enthält Hilfsskripte u. Ä. um bspw. die Daten vorzubereiten oder die Plots zu generieren
    - `dashboard.py`: Visualisierung der Ergebnisse in einem Dashboard im Browser, kann vom Rootverzeichnus aus mit `streamlit run util/dashboard.py` gestartet werden. Weitere Details im Abschnitt [Dashboard](#dashboard)
- `model_loader.py`: Enthält die Logik zum Laden der Modelle und eine Config mit Standardpfaden und die Bezeichner der Modelle
- `test.py`: Enthält die Testpipeline
- `train.py`: Enthält die Trainingspipeline

## trainierte Modelle

Die trainierten Modelle können [hier](https://1drv.ms/f/c/347f433a03da65d9/En9VDC_8_yRJk9J0XyLEYDYBDQG1nG6RQZe8h7XNQLV5Dw?e=pctfJ1) aufgerufen werden, da sie für GitHub zu groß waren. Sie müssen dann im root-Verzeichnis des Projekts in den Ordner `checkpoints/` abgelegt werden

## Dashboard 

Für jeden Tab kann am linken Rand das verwendete Modell ausgewählt werden.

Im 1. Tab werden die Trainingsergebnisse zusammengefasst und verwendeten Ressourcen angezeigt. Über ein Dropdown kann das entsprechende Trainigsszenario ausgewählt werden, entsprechend der oben genannten Test- und Trainingsdaten.

Im 2. Tab werden die Ergebnisse der Trainings dargestellt. Hier kann ebenfalls das Trainingsszenario ausgewählt werden. 

Im 3. Tab werden die Ergebnisse der Testszenarien dargestellt. Hier kann neben dem Testszenario auch die Augmentierungsart ausgewählt werden. 

Im 4. Tab sind die Delta-Plots zum Vergleich der Performance der Modelle bei Augmentierung zu finden. Auch hier kann das Testszenario ausgewählt werden.

Im 5. Tab sind die Grad-CAM Visualisierungen sortiert nach Klassifizierungsergebnis der Modelle zu finden. Hier kann das Testszenario und die Augmentierungsart ausgewählt werden.

Im 6. Tab können Plots zum Test zusammengeklickt werden. Dieser Tab wurde allerdings nicht weiter verfolgt, da keine weiteren Plots, über die bereits vorhandenen hinaus, nötig waren. Entsprechend ist dieser Tab auch noch eher experimentell von einer ersten Implementierung und nicht weiter fertiggestellt worden.

Im 7. Tab können Plots zum Training zusammengeklickt werden. Auch dieser Tab ist nicht komplett fertiggestellt, da es erstmal gereicht hatte mit diesem Stand zu arbeiten.