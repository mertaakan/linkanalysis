# Teknik Proje Raporu: Yönlü Web Graflarında PageRank ve HITS Analizi

## 1. Amaç ve Kapsam
Bu projede yönlü web graflarında düğümlerin önem sıralamasını hesaplayan **PageRank** ve **HITS** (Hub ve Authority) algoritmaları `NumPy` kullanılarak sıfırdan uygulanmıştır. Sentetik ağ yapıları (`normal`, `spider_trap`, `dead_end`) üzerinde algoritmaların davranışları incelenmiş, sonuçlar `NetworkX` kütüphanesi ile doğrulanmış ve etkileşimli bir **Streamlit** simülatörü geliştirilmiştir.

## 2. Proje Mimarisi
- `src/graph.py`: Sentetik yönlü web graflarını (`normal`, `spider_trap`, `dead_end`) oluşturan ve `NetworkX` dönüşümünü sağlayan `WebGraph` sınıfı.
- `src/pagerank.py`: Güç yöntemi (Power Iteration), teleportation ve ölü yol (dead end) düzeltmesi içeren `PageRank` sınıfı.
- `src/hits.py`: $L_2$ birim normu ile Hub ve Authority skorlarını hesaplayan `HITS` sınıfı.
- `src/experiments.py`: Deneyleri çalıştırıp grafikleri `report/figures/` klasörüne kaydeden betik.
- `tests/test_algorithms.py`: Algoritmaların doğruluğunu ve `NetworkX` uyumunu kontrol eden 6 adet `pytest` birim testi.
- `app.py`: Parametrelerin canlı değiştirilebildiği Streamlit simülatör arayüzü.

## 3. Yöntem ve Matematiksel Formüller

### 3.1. PageRank
Geçiş matrisi $M$ ve sönümleme faktörü $\alpha \in [0, 1]$ olmak üzere her iterasyonda skor vektörü şu şekilde güncellenir:

$$r^{(t+1)} = \alpha M r^{(t)} + \frac{1 - \alpha}{n}\mathbf{1}$$

Çıkış bağlantısı olmayan (`dead end`) düğümlerin sütunları $1/n$ ile doldurularak olasılık sızıntısı engellenir. Ardışık iterasyonlar arasındaki $L_1$ norm farkı (`||r_next - r||_1 < tol`) yakınsama kriteri olarak alınmıştır.

### 3.2. HITS
Komşuluk matrisi $A$ kullanılarak her adımda otorite ($a$) ve rehber ($h$) vektörleri güncellenir ve sayısal kararlılık için $L_2$ normuna bölünerek normalize edilir:

$$a \leftarrow \frac{A^T h}{\|A^T h\|_2}, \quad h \leftarrow \frac{A a}{\|A a\|_2}$$

## 4. Deneysel Bulgular ve Analiz

### 4.1. Sönümleme Faktörü ($\alpha$) ve Yakınsama Hızı
Normal graf üzerinde $\alpha = 0.50$ için 14, $\alpha = 0.85$ için 22 ve $\alpha = 0.99$ için 27 iterasyonda yakınsama sağlanmıştır (`report/figures/alpha_convergence.png`). $\alpha$ arttıkça iterasyon sayısının artması beklenen eğilimdir. $\alpha = 0.99$ değerinde yavaşlamanın çok sert olmamasının sebebi, yakınsama hızının tek başına $\alpha$'ya değil $\alpha \cdot |\lambda_2|$ çarpımına bağlı olmasıdır. Test edilen graf küçük olduğundan ikinci en büyük özdeğer $|\lambda_2|$ küçüktür; daha büyük ve seyrek web graflarında $\alpha = 0.99$ için iterasyon farkı çok daha belirgin hale gelir.

### 4.2. Spider Trap ve Periyodiklik
Spider Trap grafında `D4` ve `D5` düğümleri kapalı bir döngü (`4 ↔ 5`) oluşturur (`report/figures/spider_trap.png`). Teleportation kapalıyken tüm skor bu iki düğümde toplanır. Ayrıca `4 ↔ 5` döngüsü periyodik olduğu için zincir sabit bir denge noktasına oturmaz; skorlar iki durum arasında salınır (`D4 ≈ 0.45`, `D5 ≈ 0.55`). Teleportation ($\alpha = 0.85$) açıldığında hem tuzak etkisi kırılır hem de Markov zinciri **aperiyodik** hale gelerek kararlı bir dağılıma yakınsar.

### 4.3. Dead End (Ölü Yol) Skor Sızıntısı
`D5` düğümünün çıkış bağlantısı olmadığı için çözüm uygulanmadığında ilk iterasyonda toplam skor `0.833`'e düşer ve ilerleyen iterasyonlarda `0`'a yaklaşır (`report/figures/dead_end.png`, çözümsüz çubuklar ≈ 0). Ölü yol düzeltmesi ve teleportation birlikte uygulandığında toplam skor her iterasyonda `1.0` olarak korunur.

### 4.4. HITS Sonuçları ve NetworkX Doğrulaması
Normal grafta HITS çıktılarında `D0` ve `D5` için Authority skoru `0`, `D2/D4/D5` için Hub skoru `0` olarak hesaplanmıştır (`report/figures/hits_scores.png`). Bu durum bir uygulama hatası değil, HITS algoritmasının bilinen bir özelliğidir: Authority vektörü $A^T A$ matrisinin baskın özvektörüdür ve bağlantısız bileşenlerden baskın özdeğere sahip olan grupta (`{1, 2, 3, 4}`) toplanarak diğer bileşende sıfır skor üretebilir. `NetworkX` ile yapılan karşılaştırmada PageRank için maksimum fark `6.67e-12`, HITS Hub için `2.09e-10` ve Authority için `2.96e-10` bulunmuştur.

## 5. Testler ve Arayüz
`pytest -v` ile çalıştırılan 6 birim testinin tamamı başarılıdır. `streamlit run app.py` ile başlatılan arayüz üzerinden farklı graf türleri ve $\alpha$ değerleri etkileşimli olarak gözlemlenebilmektedir.
