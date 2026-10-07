# AI-EKG skorlarının kısa aralıklı test–tekrar test güvenilirliği: analiz protokolü

**Sürüm:** 1.0 — 5 Ekim 2026. Bu dosya skorlar hesaplanmadan önce yazıldı.
Sonradan yapılan her değişiklik aşağıdaki "Değişiklik kaydı" bölümüne
gerekçesiyle eklenecek.

## Soru

Gerçek değeri saatler içinde değişmeyen hedefler (kronolojik yaş, kayıtlı
cinsiyet) için eğitilmiş AI-EKG modellerinin çıktısı, aynı hastada dakikalar
ya da saatler arayla çekilen iki EKG arasında ne kadar değişiyor? Bu değişim
hangi çekim ve sinyal koşullarıyla ilişkili?

Hedef sabit olduğu için iki kayıt arasındaki skor farkı, kardiyak kaynaktaki
kısa süreli değişimin ve çekim ile ölçümden gelen değişimin toplamıdır. Yaş
modelinde "gerçek" değişim pratikte sıfırdır. Bu nedenle fark doğrudan ölçüm
hatası olarak yorumlanabilir.

## Veri ve kohortlar

| Kohort | Hastalar | Not |
| --- | --- | --- |
| HEEDB I0001 test | `split='test'` hastaları | Modeller yalnız `train` hastalarında eğitilir; model seçimi `val` hastalarında yapılır |
| HEEDB I0006 | Tüm hastalar | Bağımsız kurum |
| MIMIC-IV-ECG | Tüm hastalar | HEEDB modelleri ve yayımlanmış model için tamamen dış veri; cihaz, filtre, ritim ve QRS süresi makine ölçümlerinden alınır |
| MIMIC acil, troponin-negatif ve kararlı | Mevcut `stable_primary` çiftleri | Klinik olarak sınırlandırılmış alt kohort |

## Çift tanımı

- Bir hastanın zamana göre sıralanmış EKG'lerinde **ardışık** iki kayıt.
- Aralık katmanları: <5 dk, 5–60 dk, 1–6 sa (**birincil**), 6–24 sa,
  1–7 gün, 7–30 gün, 30 gün–1 yıl, >1 yıl.
- Her hasta her katmana en fazla bir çiftle katılır. Seçim, kayıt
  kimliklerinin deterministik hash'iyle yapılır.
- Her katmanda en fazla 25.000 hasta alınır (hasta hash'iyle örneklem).
- Dışlamalar: eşit zaman damgası; 12 derivasyon, 10 sn ve mV birimi
  dışındaki kayıtlar; sonlu olmayan değer, düz derivasyon (SD < 0,005 mV),
  aşırı genlik (> 20 mV); **kopya dalga formu** (iki kayıt arasındaki tüm
  derivasyonlarda en büyük mutlak fark < 0,01 mV). Kopyalar sayılır ve
  raporlanır.
- Ritim düzensizliği dışlama nedeni değildir. RR değişkenlik katsayısı ve
  kalp hızı ortak değişken olarak kullanılır. Yalnız düzenli ritim
  (RR CV ≤ 0,15) duyarlılık analizidir.

## Modeller

1. **Yayımlanmış ECG-yaş modeli** (Lima ve ark., *Nat Commun* 2021;
   ağırlıklar Zenodo 10.5281/zenodo.4892365, CC-BY-4.0). İnce ayar yapılmaz.
   Girdi 400 Hz, 4096 örnek; 4000 örnek iki yana eşit sıfırla doldurulur.
   Genlik çarpanı {1, 10} arasından **I0001 doğrulama hastalarında** en
   düşük ortalama mutlak hatayı veren değer olarak seçilir.
2. **HEEDB yaş modeli:** mevcut `ECGSexNet` mimarisi, tek çıktı, L1 kaybı;
   I0001 eğitim hastaları (hasta başına bir EKG); 5 farklı tohum (seed).
3. **HEEDB cinsiyet modeli:** aynı mimari, `SexDSC` hedefi; I0001 eğitim
   hastaları; 5 tohum. Mevcut yalnız-ekstremite ve yalnız-prekordiyal
   modeller ikincil olarak skorlanır.

Ana analizde 5 tohumun ortalaması kullanılır. Tek tohumlu sonuçlar,
model tohumundan gelen değişkenliği ayırmak için raporlanır.

## Ölçütler

Her kohort ve aralık katmanında, hasta düzeyinde (bir çift):

- **Yaş:** iki skor arasındaki fark d = ŷ_B − ŷ_A; hasta içi SD
  S_w = SD(d)/√2; tekrarlanabilirlik katsayısı RC = 2,77·S_w;
  Bland–Altman yanlılığı ve uyum sınırları. ICC(A,1) hem tahmini yaş hem
  **yaş farkı** (ŷ − kronolojik yaş) için. Uzun aralıklarda kronolojik yaş
  değişimi farktan çıkarılır.
- **Yaş kategorisi:** yaş farkı ≥ +8 yıl (Lima ve ark. eşiği) sınıfının
  iki EKG arasında değişme oranı ve Cohen kappa.
- **Cinsiyet:** logit için S_w ve ICC(A,1); 0,5 olasılık eşiğinde sınıf
  değiştirme oranı; |Δp| > 0,2 oranı.
- Güven aralıkları: hasta düzeyinde 2.000 bootstrap.
- **Ortalama alma:** Spearman–Brown ile k EKG ortalamasının beklenen ICC'si.
  Gözlemsel kontrol olarak, aynı hastada <1 sa içinde üç EKG varsa
  üçüncü EKG'ye karşı tek EKG ve iki EKG ortalaması karşılaştırılır.

**Birincil sonuç:** 1–6 saatlik katmanda yayımlanmış yaş modelinin S_w ve
yaş farkı ICC'si; I0001 test, I0006 ve MIMIC kohortlarında.

## Belirleyiciler (ikincil, keşifsel)

|d| (yaş) ve |Δlogit| (cinsiyet) için doğrusal model, standart hatalar
hastaya göre kümelenmiş. Aday değişkenler: kalp hızı farkı, RR CV,
iki kaydın sinyal gürültüsü (yüksek frekans güç oranı ve taban kayması),
ekstremite ve prekordiyal derivasyonlarda RMS genlik değişimi, yaş ve
kayıtlı cinsiyet. MIMIC'te ek olarak: cihaz (`cart_id`) değişimi, filtre
değişimi, ritim sınıfı değişimi, pace varlığı, QRS süresi farkı.

## Yorum sınırları

- Saatler arasındaki fark yalnız ölçüm hatası değildir; kısa süreli
  fizyolojik değişim de katkı verebilir. <5 dk katmanı bu katkının en az
  olduğu, ama seçilmiş (çoğunlukla teknik nedenle tekrarlanan) bir taban
  olarak yorumlanır.
- Bu çalışma çıktının tutarlılığını ölçer, doğruluğunu ya da prognostik
  değerini ölçmez.
- Cinsiyet alanları idari kayıttır; skorlar hormon, anatomi ya da kimlik
  ölçümü değildir.

## Değişiklik kaydı

Aşağıdaki maddeler skorlar görülmeden önce, uygulama sırasında eklendi.

- **Kopya tespiti:** Tam sinyal yerine iki ölçüt kullanıldı: 0,005 mV'a
  nicemlenmiş 250 Hz sinyalin MD5 özeti eşitliği veya 10 örneklik (40 ms)
  blok ortalamalarında en büyük mutlak fark < 0,01 mV. Gerekçe: 949 bin
  kaydın tam sinyalini çift başına yeniden okumamak.
- **Yaş analizlerinde** ilk EKG'deki kronolojik yaş 18 ile 89 arasında
  (89 hariç) olmalı. Gerekçe: HEEDB ve MIMIC 89 yaş üstünü sansürlüyor.
- **Model eğitimi:** HEEDB yaş ve cinsiyet modelleri, I0001 eğitim
  bölümünden rastgele seçilen 300.000 erişkin hastada (hasta başına bir
  EKG, cinsiyet etiketi tutarlı) eğitildi. 20.000 doğrulama hastasında
  en iyi epoch seçildi. 12 epoch, OneCycle öğrenme oranı. Doğrulama sonuçları:
  yaş MAE 7,65–7,67 yıl, cinsiyet AUC 0,960–0,961.
- **Lima modeli genlik çarpanı:** 5.000 I0001 doğrulama hastasında ×1
  (MAE 10,4 yıl, r = 0,73) ×10'dan (MAE 14,7 yıl) iyi olduğu için ×1 seçildi.
- **Üçlü analizi:** "üç EKG <1 sa" yerine "ardışık üç EKG, toplam süre
  <6 sa" kullanıldı. Bu ölçüt, ortalama almanın üçüncü EKG ile uyumu
  beklenen oranda (√0,75 ≈ 0,87) artırıp artırmadığını sınar. Hataların
  oturum içinde korelasyonlu olup olmadığı ise kısa ve uzun aralık S_w
  karşılaştırmasıyla değerlendirilir.
- **Ek ölçüt:** aynı EKG'de 5 tohum arasındaki SD (yalnız model kaynaklı
  değişkenlik) EKG'ler arası değişkenlikle karşılaştırılır.

**Sonuçlar görüldükten sonra eklenen (post-hoc) analizler.** Bunlar
keşifseldir; makalede böyle etiketlenecektir.

- <5 dk ile 5–60 dk arasında S_w'deki sıçramayı incelemek için:
  (a) aralık katmanına göre medyan sinyal değişimi (prekordiyal ve
  ekstremite RMS log-oranı, kalp hızı farkı); (b) 1–6 sa çiftlerinde
  prekordiyal genlik değişiminin beşte birlik dilimlerine göre S_w;
  (c) üç sinyal değişim ölçüsünün üçü de <5 dk medyanının altında
  kalan 1–6 sa çiftlerinde S_w.
- Cinsiyet ölçütleri yalnız erişkinlerde (ilk EKG'de ≥18 yaş) yeniden
  hesaplandı, çünkü modeller erişkinlerde eğitildi. Birincil analiz
  protokoldeki gibi tüm yaşları kapsar.
- **Artefakt ve ters takılma duyarlılığı** (kullanıcı sorusu üzerine,
  6 Ekim 2026; `reliability_artifact_check.py`):
  (a) MIMIC makine raporunda "lead reversal", "unsuitable for analysis",
  "please repeat" ya da "external noise" geçen EKG'yi içeren çiftler
  çıkarıldı; (b) tüm kohortlarda, iki EKG'den biri sitenin yüksek frekans
  gürültüsü ya da taban kayması dağılımının en üst %10'unda olan çiftler
  çıkarıldı. HEEDB'de makine raporu yok; sinyale dayalı ters takılma
  tespiti henüz yapılmadı.
- **Sinyale dayalı ters takılma tespiti** (kullanıcı isteğiyle, 6 Ekim 2026;
  `reliability_reversal_check.py`): B EKG'sinin ortanca atımına her
  ekstremite elektrot permütasyonu (Einthoven ilişkileriyle I/II üzerinden)
  ve komşu prekordiyal takas uygulanır. Bir permütasyon A'ya uyumu
  özdeşlikten belirgin biçimde artırıyorsa çift işaretlenir. Eşikler
  (ekstremite oranı < 0,8; prekordiyal oranı < 0,7) MIMIC makine
  raporlarına karşı seçildi ve HEEDB'ye değiştirilmeden uygulandı.
  Tek EKG kuralı (DI negatif, aVR pozitif QRS) MIMIC'te kesinliği %2,3
  olduğu için kullanılmadı.
- **HEEDB 12SL makine yorumları** (6 Ekim 2026;
  `reliability_heedb_12sl_check.py`): 12SL v24 kodları kullanıldı.
  Ters takılma: 1672, 1595, 1679. Kalite: 1302, 1500, 1501, 1502, 1504,
  1673. Bayraklı çiftler çıkarılarak S_w yeniden hesaplandı ve sinyal
  detektörü bu kodlara karşı doğrulandı.
- **HEEDB I0006 kayıt sayısı:** HEEDB *Scientific Data* 2026 makalesi
  Emory için 998.844 kayıt bildiriyor. Bu, yerel dosyayla aynı; önceki
  raporda "eksik alt küme" olarak belirtilen fark gerçek bir eksiklik
  değil.
- **Yanlılık düzeltilmiş yaş farkı** (Barthels ve ark. 2025): her site ve
  model için yaş farkı, ilk EKG'lerde kronolojik yaşa göre doğrusal
  regresyonla düzeltildi; >8 yıl sınıfı bu artık üzerinden yeniden
  hesaplandı.
- **İkinci ve üçüncü yayımlanmış model** (kullanıcı isteğiyle, 6 Ekim
  2026; tanımlar sonuçlar görülmeden önce yazıldı):
  - **Bracke ve ark., MICCAI 2026** (`reliability_bracke.py`): cinsiyetle
    koşullandırılmış BiMamba2 ECG-yaş modeli; CODE-15% ile eğitilmiş;
    Hugging Face ağırlıkları, MIT. Girdi Lima ile aynı (400 Hz, 4096
    örnek, simetrik sıfır doldurma); model her derivasyonu kendi içinde
    normalize eder. Kayıtlı cinsiyet girdi olarak verilir; cinsiyeti
    bilinmeyen ya da çelişkili EKG'ler skorlanmaz. mamba-ssm CUDA eklentisi
    olmadan kuruldu; Mamba2 aynı işlemi yapan Triton yolu
    (`use_mem_eff_path=False`) ile çalıştırıldı. Yaş analizlerinin tamamı
    (aralık eğrisi, yanlılık düzeltmesi, üçlüler, belirleyiciler) bu model
    için de tekrarlanır.
  - **Ribeiro ve ark., Nat Commun 2020** (`reliability_ribeiro.py`,
    `reliability_diagnostic_analysis.py`): 6 anormallik modeli (CODE ile
    eğitilmiş; Zenodo, CC-BY-4.0). Genlik çarpanı {1, 10}, 5.000 I0001
    doğrulama EKG'sinde 12SL etiketlerine karşı ortalama AUC ile seçildi
    (×1: 0,983). Birincil sınıflar: 1. derece AV blok, RBBB, LBBB.
    Ölçütler: 0,5 eşiğinde sınıf değişimi ve kappa; ilk EKG'de pozitif
    olanlarda ikinci EKG'de kayıp oranı; logit S_w ve ICC. Aynı çiftlerde
    makine yorumu (HEEDB 12SL v24, MIMIC makine raporu) kıyaslama olarak
    raporlanır.
- **Kod düzenlemesi:** torch'a bağımlı olmayan WFDB okuma fonksiyonları
  `ecg_io.py` dosyasına taşındı; davranış değişmedi.
