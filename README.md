# PageRank ve HITS Link Analizi Projesi

Yonlu web graflari uzerinde **PageRank** ve **HITS** (Hub ve Authority) algoritmalarinin sifirdan (`NumPy` ile) uygulanmasi, sonumleme faktoru (alpha), Spider Trap ve Dead End senaryolarinin deneysel analizi, birim testleri ve interaktif **Streamlit** simulator arayuzunu iceren calismadir.

## Proje Yapisi
```
linkanalysis/
├── src/
│   ├── graph.py           # Sentetik web grafi ureticileri ve matris yardimcilari
│   ├── pagerank.py        # Teleportation ve dead-end destekli PageRank sinifi
│   ├── hits.py            # L2 birim normlu Hub ve Authority hesaplayan HITS sinifi
│   └── experiments.py     # Deneysel senaryolar ve grafik uretim betigi
├── tests/
│   └── test_algorithms.py # pytest birim testleri
├── report/
│   ├── figures/           # Uretilen deney grafikleri (.png)
│   └── technical_report.md
├── app.py                 # Interaktif Streamlit link analizi simulatoru
├── pytest.ini             # pytest yol ayari
└── requirements.txt
```
## Kurulum
```
cd ~/Desktop/linkanalysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Kullanim
```
1. Deneyleri ve grafikleri calistirmak icin:
   python -m src.experiments
```
```
2. Birim testlerini calistirmak icin:
   pytest -v
```
```
3. Streamlit simulatorunu baslatmak icin:
   streamlit run app.py
```
