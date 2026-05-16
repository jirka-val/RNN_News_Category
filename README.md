# NLP Projekt: Pokročilá klasifikace zpráv pomocí RNN a Transfer Learningu

Tento projekt se věnuje problematice zpracování přirozeného jazyka (NLP) se zaměřením na úlohu vícetřídní klasifikace textových dat (Multi-class Text Classification). Cílem projektu je vývoj, optimalizace a exaktní komparace hlubokých rekurentních neuronových sítí pro automatickou kategorizaci zpravodajských článků.

## Charakteristika Datové Sady a Metrika

V projektu je využit **News Category Dataset** obsahující zpravodajské záznamy z let 2012 až 2022. Hlavní výzvou této datové sady je její **extrémní třídní nevyváženost (Class Imbalance)**. Dominantní kategorie (např. *POLITICS*) obsahují až 35 000 vzorků, zatímco minoritní kategorie (např. *EDUCATION*) disponují pouze přibližně 1 000 vzorky.

Z tohoto důvodu byla jako hlavní evaluační a rozhodovací metrika projektu zvolena hodnota **Macro F1-score**. Na rozdíl od globální přesnosti (Accuracy) tato metrika vypočítává prostý průměr F1-score napříč všemi 15 třídami nezávisle na jejich velikosti, čímž model penalizuje za ignorování minoritních kategorií.

---

## Fáze 1: Model od nuly (Model from Scratch)

První fáze projektu spočívala v návrhu vlastní architektury, kde se vrstva embeddingu učila reprezentaci slov výhradně na trénovacích datech zadaného korpusu. Vývoj probíhal iterativně pomocí tří experimentů:

* **Experiment A (Základní Baseline):** Použití jednosměrné vrstvy LSTM (32 jednotek) a nízké dimenze embeddingu (50) bez jakékoliv regularizace. Výsledkem byl velmi rychlý nástup přeučení (overfittingu).
* **Experiment B (Regularizace):** Zařazení vrstev `Dropout` (0.3) a `BatchNormalization` za účelem stabilizace vah a zpomalení divergence validační chybové funkce.
* **Experiment C (Obousměrný kontext):** Nasazení vrstvy `Bidirectional(LSTM)` se 64 jednotkami pro každý směr a NLP-specifické regularizace `SpatialDropout1D`. Schopnost sítě analyzovat text v obou směrech výrazně zlepšila sémantické porozumění.

**Finální Model 1:** Architektura z Experimentu C byla odtrénována na plný počet 20 epoch bez mechanismu Early Stopping.
* **Testovací Accuracy:** 0.7093
* **Testovací Macro F1-Score:** 0.6322

---

## Fáze 2: Transfer Learning (Předtrénované vektory GloVe)

Druhá fáze testovala přínos přenosu externích znalostí pomocí předtrénovaných slovních vektorů **GloVe (Global Vectors for Word Representation)** od Stanford University (100-dimenzionální varianta). Postup byl opět rozdělen do tří experimentů:

* **Experiment A (Statický GloVe):** Inicializace embeddingu vahami GloVe se striktním zmrazením vrstvy (`trainable=False`) v kombinaci s jednoduchou LSTM síti.
* **Experiment B (Komplexní statická struktura):** Použití zmrazeného embeddingu GloVe v kombinaci s obousměrnou LSTM a regularizačním blokem. Výsledky ukázaly, že statické vektory tvoří úzké hrdlo (bottleneck), protože jejich obecný význam plně neodpovídá specifickému žargonu internetových zpráv.
* **Experiment C (Jemné doladění / Fine-Tuning):** Odemčení embedding vrstvy pro trénování (`trainable=True`) se sníženým learning rate (0.0005). Tento přístup umožnil síti adaptovat bohaté sémantické základy GloVe přímo na kontext zadaného datasetu a přinesl skokový nárůst výkonu.

**Finální Model 2:** Architektura z Experimentu C (Fine-Tuning) byla odtrénována na plný počet 20 epoch bez Early Stopping.
* **Testovací Accuracy:** 0.7508
* **Testovací Macro F1-Score:** 0.6846

---

## Závěrečné Kvantitativní Srovnání

| Metrika na testovacích datech | Model 1 (Vlastní od nuly) | Model 2 (Transfer Learning + Fine-Tuning) | Absolutní rozdíl |
| :--- | :---: | :---: | :---: |
| **Testovací Accuracy** | 0.7093 | 0.7508 | **+4.15 %** |
| **Testovací Macro F1-Score** | 0.6322 | 0.6846 | **+5.24 %** |

### Hlavní Závěry
1.  **Dominance Transfer Learningu:** Implementace předtrénovaných vektorů v kombinaci s jemným doladěním (Model 2) jednoznačně překonala trénování sítě z nuly.
2.  **Robustnost vůči nevyváženosti:** Výrazný nárůst klíčové metriky **Macro F1-score o více než 5.2 %** prokazuje, že externí znalost pomohla modelu lépe generalizovat a úspěšně klasifikovat minoritní, podreprezentované třídy zpravodajství, u kterých měl baseline model nedostatek vlastních dat.
3.  **Význam Fine-Tuningu:** Pouhé nasazení statických vektorů nestačí. Klíčem k úspěchu je možnost adaptace embeddingové matice na doménová specifika konkrétní úlohy.

## Požadavky na Prostředí
* pandas>=2.0.0
* numpy>=1.24.0
* matplotlib>=3.7.0
* seaborn>=0.12.0
* scikit-learn>=1.2.0
* tensorflow>=2.13.0
* kagglehub>=0.1.0
* ipywidgets>=7.0.0
