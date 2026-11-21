# TR-LLM Bench

[English](README.md) · **Türkçe**

Açık LLM'lerde Türkçe servis etmenin maliyeti nedir ve quantization, İngilizceye kıyasla
Türkçede ne kadar kalite kaybına yol açar?

**[Leaderboard](https://huggingface.co/spaces/HaticeB/tr-llm-bench)** · **[Eval seti](https://huggingface.co/datasets/HaticeB/tr-llm-bench-eval)**

Buradaki her şey paralel TR/EN veri üzerinde, deterministik puanlamayla (seed 2026, LLM judge
yok) ölçüldü. Bu yüzden Türkçe ve İngilizce sonuçlar doğrudan karşılaştırılabilir.

## 1. Tokenizer vergisi

Aynı FLORES+ cümleleri, dile göre farklı sayıda token'a bölünür.
**Vergi = toplam Türkçe token / toplam İngilizce token.** Kelime uzunlukları diller arasında
farklı olduğu için fertility (kelime başına token) kullanılmadı.

| Tokenizer (onu paylaşan modeller) | TR | FI | HU | DE | FR |
|---|---|---|---|---|---|
| Kumru-2B (Türkçe odaklı) | 0.64 | 1.49 | 1.69 | 1.48 | 1.52 |
| Trendyol-7B | 1.10–1.22 ¹ | 1.95 | 1.94 | 1.56 | 1.58 |
| Aya-Expanse-8B | 1.21 | 1.93 | 2.08 | 1.32 | 1.33 |
| Gemma-3-12B | 1.36 | 1.66 | 1.67 | 1.32 | 1.40 |
| Gemma-2-9B | 1.40 | 1.61 | 1.66 | 1.25 | 1.35 |
| Llama-3.1-8B / Cosmos-TR-Llama-8B | 1.40 | 1.98 | 2.11 | 1.58 | 1.59 |
| gpt-oss-20b / Phi-4-mini | 1.42 | 1.56 | 1.79 | 1.31 | 1.36 |
| Mistral-Nemo-12B / Ministral-8B | 1.48 | 1.65 | 1.61 | 1.30 | 1.31 |
| Qwen2.5-7B / Qwen3-8B | 1.62 | 1.97 | 2.11 | 1.55 | 1.57 |
| Granite-3.3-8B | 1.72 | 1.87 | 1.91 | 1.61 | 1.62 |
| Phi-4 | 1.92 | 1.99 | 2.14 | 1.58 | 1.59 |
| DeepSeek-V3 | 2.03 | 1.87 | 1.98 | 1.51 | 1.54 |
| Mistral-7B-v0.3 | 2.21 | 2.01 | 2.00 | 1.58 | 1.59 |

Aynı 2.009 cümlede İngilizceye göre token sayısı (İngilizce = 1.0). %95 bootstrap aralıkları
(cümleler yeniden örneklenerek, diller arasında eşleştirilmiş) `results/tokenizer_langs.csv`
dosyasında; cümle başına token sayıları (yalnızca sayılar, metin yok) `results/tokenizer_counts.csv.gz` dosyasında.

1.62'lik bir vergi, aynı Türkçe içeriğin %62 daha fazla token tuttuğu anlamına gelir: daha fazla
latency, daha az kullanılabilir context ve daha düşük içerik throughput'u. Trendyol'un vocabulary'ye
Türkçe token eklemesi (32.768 → 44.312 token) vergiyi 1.10'a, genel modeller arasındaki en düşük
değere indiriyor.

**Maliyet Türkçeye özgü değil.** 13 farklı tokenizer'ın 11'inde Türkçe, diğer iki eklemeli dil
olan Fince ve Macarcadan daha az token tutuyor (aralıklar çakışmıyor); istisnalar Mistral-7B-v0.3
ve DeepSeek-V3. Almanca ve Fransızcaya karşı tablo karışık. Bu, sorunun yalnızca morfolojide değil
tokenizer'ın eğitim verisinde olduğunu gösteriyor: Türkçe odaklı bir tokenizer (Kumru) Türkçeyi
İngilizceden bile *ucuz* hale getiriyor.

¹ *Aynı tokenizer dosyaları (repo Haziran 2024'ten beri değişmedi), farklı kütüphaneler:*
transformers 4.57.3'teki fast tokenizer 1.10, SentencePiece (slow) tokenizer 1.12 veriyor; daha eski
bir transformers sürümüyle yapılan ilk ölçüm 1.22 vermişti. Eval prompt'larında, modeli 2. bölümde
servis eden llama.cpp'nin tokenizer'ı 1.10 verirken sunucudaki transformers aynı metin için 1.16
veriyor; bu yüzden servis sonuçlarında 1.10 kullanıldı. Artık her tokenizer satırına repo commit'i,
tokenizer sınıfı ve kütüphane sürümleri kaydediliyor. Diğer bütün tokenizer'lar birebir aynı sonucu verdi.

![Türkçe ve diğer diller](docs/tokenizer_langs.png)

![Tokenizer vergisi](docs/tokenizer_tax_en.png)

## 2. Paylaşımlı bir V100'de servis performansı

Yük testlerinde çıktı uzunluğu sabit (`min_tokens` + `ignore_eos`, 256 token) ve TTFT streaming ile
ölçülüyor. Türkçe token'lar İngilizce token'lardan daha az içerik taşıdığı için ham tok/s,
**içerik throughput'u = tok/s ÷ vergi** ile birlikte raporlanıyor.

GPU bir production servisiyle paylaşılıyordu. Benchmark tek bir V100'de yaklaşık 8 GB'lık bir
bütçeyle, diğer GPU'lar ya da servis etkilenirse ölçümü durduran bir guard altında çalıştı.
O GPU'daki toplam bellek kullanımı, servis dahil, en fazla 12.9 GB'ta kaldı (planlanan tavan 15.5 GB).
138 koşu, 0 hata.

### 2a. Yedi model, llama.cpp, Q4_K_M

3 tekrarın medyanı. Maliyet, tam yükte **GPU saati başına 1 $** varsayıyor (kendi fiyatınıza göre
doğrusal ölçekleyin); saniyedeki Türkçe kelime = token ÷ FLORES+ üzerindeki kelime başına Türkçe token.

| Model | c=1 TR tok/s | En yüksek c | TR tok/s | EN tok/s | TR içerik tok/s | TTFT p50 TR (ms) | GPU 2 tepe (GB) | 1 $/saatte 1.000 TR kelime başına $ |
|---|---|---|---|---|---|---|---|---|
| Trendyol-7B | 35 | 8 | 130 | 123 | **119** | 427 | 10.3 | 0.0040 |
| Aya-Expanse-8B | 32 | 8 | 109 | 115 | **90** | 571 | 11.6 | 0.0047 |
| Cosmos-TR-Llama-8B | 34 | 8 | 122 | 126 | **87** | 493 | 11.3 | 0.0049 |
| Llama-3.1-8B | 34 | 8 | 121 | 123 | **87** | 636 | 12.4 | 0.0049 |
| Qwen2.5-7B | 34 | 8 | 130 | 130 | **81** | 249 | 9.9 | 0.0054 |
| Mistral-7B-v0.3 | 35 | 8 | 127 | 128 | **57** | 683 | 11.0 | 0.0082 |
| Gemma-2-9B | 25 | 2 | 43 | 42 | **31** | 531 | 12.9 | 0.0139 |

![GPU'da saniyedeki Türkçe içerik](docs/gpu_content.png)

- **Aynı engine, aynı GPU, benzer ham hız (8 eşzamanlı istekte 121–130 tok/s); yine de saniyedeki
  Türkçe içerik 57 ile 119 arasında değişiyor.** Bir GPU saatinin ne kadar Türkçe ürettiğini model
  boyutu değil tokenizer belirliyor: Mistral-7B'de Türkçe kelime başına maliyet Trendyol'un yaklaşık iki katı.
- Gemma-2-9B 2 eşzamanlı istekle çalışıyor: KV cache'i (token başına ~344 KB) bütçeye 8 istek sığdırmıyor.
- Trendyol'un vergisi 1.10 alındı: FLORES+ üzerinde fast tokenizer'ın verdiği değer; modeli servis
  eden llama.cpp'nin kendi tokenizer'ı da eval prompt'larında bu değeri veriyor (1. bölümdeki ¹ notuna bakın).

### 2b. Engine karşılaştırması: vLLM ve llama.cpp (Mistral-7B, 4-bit)

Eager modda AWQ-int4 ile vLLM (CUDA graph'lar bütçeden fazla bellek istediği için kapalı) ve
Q4_K_M ile llama.cpp; aynı model, aynı GPU, aynı bütçe.

| Eşzamanlılık | Dil | llama.cpp tok/s | vLLM tok/s | vLLM / llama.cpp | TTFT p50 llama.cpp / vLLM (ms) |
|---|---|---|---|---|---|
| 1 | EN | 36 | 26 | -%29 | 89 / 100 |
| 1 | TR | 35 | 28 | -%20 | 117 / 98 |
| 4 | EN | 85 | 100 | +%18 | 218 / 170 |
| 4 | TR | 76 | 101 | +%33 | 328 / 189 |
| 8 | EN | 128 | 199 | +%56 | 560 / 248 |
| 8 | TR | 127 | 152 | +%20 | 683 / 2482 |

- **Tek kullanıcıda llama.cpp, yük altında vLLM önde** (4 istekte Türkçede +%33, 8 istekte
  İngilizcede +%56): CUDA graph olmadan vLLM her adımda bir ek yük ödüyor ve bu yükü ancak batching dengeliyor.
- **Dar bütçede Türkçe, 8 istekte vLLM'in avantajını kaybediyor.** İngilizce 0.25 sn'lik ilk token
  ile 199 tok/s'ye ulaşıyor; Türkçe 152 tok/s'ye düşüyor ve ilk token 2.5 sn sürüyor. Muhtemel sebep:
  Mistral Türkçeyi 2.2 kat daha fazla token'a bölüyor, aynı istekler daha fazla KV cache bloğu istiyor
  ve yalnızca ~8 GB ile vLLM bunları sıraya koymak zorunda kalıyor. Tokenizer vergisi bekleme süresi olarak ortaya çıkıyor.
- Yalnızca Mistral karşılaştırılabildi: Llama-3.1 ve Qwen2.5 AWQ bütçe içinde başlamadı. 128k–152k'lık
  vocabulary'leri büyük embedding ve çıktı katmanlarını 16-bit tutuyor; yükleme 6.5 GiB aldı ve KV cache'e
  yer kalmadı (vLLM −1.01 GiB kullanılabilir bellek bildirdi).

### 2c. GPU'da context uzunluğu (llama-bench)

Her hücre 3 tekrarın medyanı. GPU paylaşımlı olduğu için tek tek tekrarlarda zaman zaman ani
düşüşler oldu (ör. 37, 37, 21 tok/s); medyan bunları sonuca taşımıyor.

| Model | Decode tok/s, boş context | 4.096 token context'te | Değişim | Prefill tok/s (4.096 token prompt) |
|---|---|---|---|---|
| Aya-Expanse-8B | 33.1 | 30.9 | -%7 | 702 |
| Llama-3.1-8B | 35.9 | 33.2 | -%8 | 704 |
| Cosmos-TR-Llama-8B | 35.9 | 33.1 | -%8 | 703 |
| Mistral-7B-v0.3 | 38.3 | 35.2 | -%8 | 705 |
| Trendyol-7B | 37.8 | 34.8 | -%8 | 706 |
| Qwen2.5-7B | 37.2 | 33.6 | -%10 | 752 |
| Gemma-2-9B | 27.1 | 23.3 | -%14 | 436 |

- Context'te 4.096 token varken decode %7–10 daha yavaş (Gemma'da %14). Türkçe aynı metin için
  1.1–2.2 kat daha fazla token gerektirdiğinden bu noktaya daha erken ulaşıyor.

### 2d. Tüm GPU'nun kullanıldığı önceki ölçüm (tarihsel)

Qwen2.5-7B-Instruct, vLLM, bir V100 32 GB, istek başına 256 çıktı token'ı, 3 tekrarın medyanı.
*Kaynak:* 16 Kasım 2026'da `configs/phase3.yaml` dosyasının önceki bir sürümüyle ölçüldü
(GPU bellek kullanımı 0.80, CUDA graph'lar açık). Bu ayar paylaşımlı sunucunun bütçesinden fazla
bellek kullandığı için config emekliye ayrıldı; bu sonuçlar güncel config'ten yeniden üretilemez ve
tarihsel bir sonuç olarak tutuluyor. İçerik tok/s = TR tok/s ÷ 1.62.

| Quant | Eşzamanlılık | EN tok/s | TR tok/s | TR içerik tok/s | TTFT p50 EN / TR (ms) |
|---|---|---|---|---|---|
| fp16 | 1 | 15.5 | 15.6 | 9.6 | 84 / 84 |
| fp16 | 4 | 61.0 | 60.6 | 37.4 | 217 / 196 |
| fp16 | 16 | 185.5 | 185.5 | 114.5 | 412 / 404 |
| fp16 | 32 | 273.8 | 265.0 | 163.6 | 540 / 602 |
| AWQ-int4 | 1 | 32.6 | 31.6 | 19.5 | 57 / 60 |
| AWQ-int4 | 4 | 124.6 | 122.7 | 75.7 | 117 / 125 |
| AWQ-int4 | 16 | 352.4 | 315.5 | 194.7 | 261 / 252 |
| AWQ-int4 | 32 | 449.9 | 431.0 | 266.0 | 529 / 554 |

- **Engine dilden bağımsız, tokenizer değil.** Ham tok/s Türkçe ve İngilizcede neredeyse aynı;
  yine de Türkçe kullanıcı saniyede ~%38 daha az içerik alıyor.
- **AWQ-int4, V100'de her eşzamanlılık seviyesinde fp16'ya göre throughput'u yaklaşık iki katına çıkarıyor**
  (tek akışlı decode bellek bant genişliğiyle sınırlı, bu yüzden 4 kat küçük ağırlıklar kazandırıyor).
- Qwen2.5-14B AWQ-int4 (yalnızca Türkçe, kısmi koşu): c=1'de 17.7 tok/s, c=32'de 203 tok/s.

Bulgu: GPTQ-int8, attention bias kullanan modellerde (ör. Qwen) V100'de (SM70) çalışmıyor, çünkü
Marlin kernel'ı bias'ı desteklemiyor.

## 3. Quantization'ın Türkçe kaliteye vergisi (llama.cpp, CPU)

550 soru: paralel TR/EN okuma (Belebele), bilgi (Global-MMLU, kültürden bağımsız alt küme),
matematik (GSM8K / GSM8K-TR), talimat takibi, JSON çıkarımı ve yalnızca Türkçe bir morfoloji seti.

**Quantization vergisi = (TR'deki doğruluk kaybı) − (EN'deki doğruluk kaybı)**, fp16'ya göre,
%95 bootstrap güven aralıklarıyla.

Yedi instruct model; GGUF'lar orijinal ağırlıklardan üretildi (Qwen: resmi GGUF), llama.cpp `b12470`,
CPU, greedy decoding. Çeviri hatalı 8 soru çıkarılmış fp16 doğruluğu (%, TR / EN; aşağıya bakın):

| Model | Okuma | Bilgi | Matematik | Talimat | JSON | Morfoloji (TR) |
|---|---|---|---|---|---|---|
| Gemma-2-9B | 87 / 95 | 63 / 66 | 97 / 97 | 98 / 85 | 100 / 100 | 88 |
| Qwen2.5-7B | 85 / 95 | 45 / 68 | 79 / 95 | 78 / 75 | 100 / 100 | 64 |
| Aya-Expanse-8B | 84 / 96 | 53 / 63 | 69 / 85 | 70 / 83 | 100 / 100 | 90 |
| Cosmos-TR-Llama-8B | 75 / 96 | 50 / 55 | 69 / 77 | 67 / 77 | 100 / 100 | 88 |
| Llama-3.1-8B | 67 / 91 | 58 / 61 | 79 / 90 | 67 / 97 | 100 / 100 | 66 |
| Trendyol-7B v1.8 | 82 / 78 | 50 / 58 | 36 / 36 | 32 / 18 | 50 / 50 | 86 |
| Mistral-7B-v0.3 | 35 / 87 | 42 / 53 | 23 / 59 | 32 / 60 | 98 / 100 | 64 |

![Quantization vergisi](docs/quant_tax.png)

242 eşleştirilmiş soru üzerinde quantization vergisi (yüzde puan, pozitif = Türkçe daha çok
kaybediyor, %95 bootstrap güven aralığı):

| Model | Q8_0 | Q4_K_M |
|---|---|---|
| Gemma-2-9B | +1.2 [−0.4, +3.3] | −0.0 [−2.5, +2.5] |
| Qwen2.5-7B | +1.2 [−1.7, +4.5] | +1.7 [−2.1, +5.4] |
| Aya-Expanse-8B | +2.1 [−0.8, +5.0] | +0.0 [−4.5, +5.0] |
| Cosmos-TR-Llama-8B | +0.8 [−2.5, +4.1] | −0.8 [−5.4, +3.7] |
| Llama-3.1-8B | −0.8 [−4.5, +2.9] | −1.7 [−7.4, +3.7] |
| Trendyol-7B v1.8 | −0.4 [−2.9, +2.1] | −2.9 [−7.4, +1.2] |
| Mistral-7B-v0.3 | −1.7 [−5.4, +1.7] | −2.5 [−7.4, +2.1] |

- **Yedi modelin hiçbirinde ölçülebilir bir quantization vergisi yok.** 14 aralığın hepsi sıfırı
  içeriyor: Q4_K_M'e kadar Türkçe, İngilizceden daha fazla bozulmuyor.
- **Diller arasındaki farkı base model belirliyor ve bu fark modelden modele çok değişiyor.**
  Gemma-2-9B matematikte eşit, bilgide yakın; Qwen2.5-7B bilgide 23 puan geride; Llama-3.1-8B okumada
  24, Türkçe talimat takibinde 30 puan geride; tokenizer vergisi en kötü model olan Mistral-7B (2.21×)
  Türkçede genel olarak en zayıf.
- **Morfolojide çok dilli ve Türkçeye fine-tune edilmiş modeller önde:** Aya 90, Gemma 88, Cosmos 88,
  Trendyol 86; Qwen, Llama ve Mistral'de 64–66.
- **Türkçe fine-tuning Türkçeye yarıyor, ama yapılandırılmış çıktıya sızabiliyor.** Trendyol-7B v1.8
  okumada Türkçede İngilizceden daha iyi olan tek model (82'ye 78), ama JSON çıktılarının yarısı geçersiz:
  prompt İngilizce olsa bile Türkçe ondalık virgül (`"birim_fiyat": 3,75`) ve Türkçe key'ler yazıyor.
  Cosmos (Türkçeye fine-tune edilmiş Llama-3) bu yan etki olmadan okumada ve morfolojide Llama-3.1-8B'yi geçiyor.
- **Yol boyunca karşılaşılan pratik tuzaklar:**
  - Trendyol-7B v1.8 (Mistral tabanlı) chat template olmadan yayımlanmış. llama.cpp sessizce ChatML'e
    geçti ve ilk koşu anlamsız çıktı; yukarıdaki sonuçlar modelin kendi `[INST]` formatıyla yapılan
    tekrar koşudan (`configs/chat_templates/mistral_inst.jinja`). Dönüştürdüğünüz her modelin template'ini
    kontrol edin; `scripts/queue_models.sh` artık template'i olmayan bir modeli ölçmüyor.
  - Aya, turn sonu token'ını (`<|END_OF_TURN_TOKEN|>`) metin olarak yazdı ve çoktan seçmeli soruları
    tek bir harf ve ardından bir açıklamayla cevapladı. Cevaplar turn token'ında kesildi (greedy decoding'de
    doğru bir durma ayarıyla aynı sonucu verir; `trbench.eval.rescore`) ve scorer, ardından satır sonu gelen
    tek bir harfi okuyacak şekilde düzeltildi. Bütün modeller aynı kurallarla yeniden puanlandı; yalnızca
    Aya'nın ve Gemma'nın 4 sorusunun sonucu değişti.

### GPU'da on dört model (Q4_K_M)

Aynı 550 soru, paylaşımlı V100'de llama.cpp (`scripts/gpu_quality.sh`), greedy decoding.
Yedi yeni model eklendi; CPU'daki yedi model aynı GGUF dosyalarıyla GPU'da yeniden çalıştırıldı.
Eşleştirilmiş sorular 8 çeviri hatası olmadan; morfoloji yalnızca Türkçe. 242 çiftle tek bir doğruluk
değerinin %95 aralığı yaklaşık ±5 puan; bu yüzden tabloda yan yana olan modeller birbirinden ayırt edilemez.

| Model | TR | EN | TR − EN | Morfoloji | CPU ile aynı sonuç |
|---|---|---|---|---|---|
| Gemma-2-9B | %90 | %89 | +1 | %88 | %99 |
| Qwen3-8B | %81 | %92 | −11 | %92 | – |
| Qwen2.5-7B | %79 | %88 | −10 | %60 | %98 |
| Qwen3-4B | %77 | %91 | −14 | %70 | – |
| Gemma-3-4B | %76 | %87 | −11 | %88 | – |
| Aya-Expanse-8B | %76 | %86 | −10 | %92 | %97 |
| Llama-3.1-8B | %74 | %85 | −11 | %44 | %95 |
| Phi-4-mini | %70 | %88 | −19 | %64 | – |
| Cosmos-TR-Llama-8B | %70 | %82 | −12 | %84 | %96 |
| Trendyol-7B | %51 | %45 | +6 | %84 | %98 |
| Llama-3.2-3B | %50 | %76 | −26 | %42 | – |
| Granite-3.3-8B | %50 | %87 | −37 | %46 | – |
| Ministral-8B | %48 | %81 | −34 | %72 | – |
| Mistral-7B-v0.3 | %48 | %71 | −23 | %58 | %94 |

- **GPU ve CPU aynı sonucu veriyor.** İki tarafta da ölçülen yedi modelde soruların %94–99'u aynı
  doğru/yanlış sonucu alıyor ve doğruluk en fazla 2 puan değişiyor: kalite sonuçları backend'e bağlı değil.
- **Türkçede belirleyici olan model boyutu değil.** Gemma-3-4B ve Qwen3-4B, 8B modellerle (Llama-3.1,
  Cosmos) aynı seviyede ya da önde; Granite-3.3-8B ve Ministral-8B ise İngilizceden Türkçeye 34–37 puan kaybediyor.
- **İngilizce doğruluk farkı gizliyor.** Granite-3.3-8B İngilizcede %87, Türkçede %50; yalnızca İngilizce
  bir benchmark onu Gemma-3-4B'nin yanına koyardı.
- İki koşunun düzeltilmesi gerekti, ikisi de kodda belgeli: Aya-Expanse turn sonu token'ını metin olarak
  yazdı (CPU'daki gibi kesilip yeniden puanlandı), Qwen3-4B ise `--reasoning-budget 0` olmasına rağmen
  düşünme adımlarını düz metin olarak yazmaya devam etti ve `enable_thinking=false` ile yeniden çalıştırıldı
  (`configs/gpu_quality_rerun.txt`). Ham satırlar: `results/gpu_quality.jsonl`;
  `python -m trbench.eval.gpu_quality_report` tabloyu yeniden üretiyor.

### Aradaki farkın ne kadarı benchmark'ın hatası?

fp16'nın Türkçede yanlış, İngilizcede doğru cevapladığı 38 sorunun hepsi elle incelendi:
**30'u gerçek model hatası, 8'i** Türkçe sorunun anlamını değiştiren **çeviri hatası**, 0'ı puanlama
hatası. Anlamı değiştiren hatalardan örnekler:

- Belebele-TR: "doğru değildir" seçeneği çeviride olumsuzlanmış, böylece doğru olmuş; "rotation"
  metinde "yörünge", seçeneklerde "dönüş" olmuş; seçeneklerde "airline" "havalimanı" olmuş.
- Global-MMLU-TR: "EXCEPT" atılmış ve soru tersine dönmüş; "H2O(s)" (katı) "H2O(lar)" olmuş (Türkçe çoğul eki).
- GSM8K-TR: "her ayakla dakikada 200 vuruş" belirsizleşmiş ("iki ayakla").

Bu 8 soru çıkarıldıktan sonra fp16'da Türkçe–İngilizce farkı:

| Kategori | Ham fark | Düzeltilmiş fark |
|---|---|---|
| Okuma | 16.7 puan | 9.0 puan |
| Bilgi | 27.5 puan | 23.7 puan |
| Matematik | 17.5 puan | 15.4 puan |

**Okumadaki farkın yaklaşık yarısı modelden değil, Belebele-TR'nin çeviri kalitesinden geliyor.**
Bilgi ve matematikteki farklar büyük ölçüde gerçek. Diğer birkaç soruda cevabı değiştirmeyen terim
hataları var (ör. "24 hours" → "24 saniye", "discount rate" → "reeskont oranı" yerine "indirim oranı").

## 4. CPU'da kalite ve hız

Her GGUF üzerinde `llama-bench` (llama.cpp `b12470`, 16 çekirdek, 128 üretilen token, 3 tekrar);
**Türkçe içerik throughput'u = tok/s ÷ tokenizer vergisi** ve eşleştirilmiş kategorilerde
Türkçe doğruluk (çeviri hatalı sorular çıkarılmış).

![Kalite ve hız](docs/pareto_cpu.png)

Her modelin en hızlı ayarı olan Q4_K_M:

| Model (Q4_K_M) | CPU tok/s | TR içerik tok/s | TR doğruluk % |
|---|---|---|---|
| **Gemma-2-9B** | 8.7 | 6.2 | 89 |
| Qwen2.5-7B | 9.2 | 5.7 | 79 |
| **Aya-Expanse-8B** | 10.9 | 9.0 | 76 |
| Llama-3.1-8B | 11.7 | 8.4 | 74 |
| Cosmos-TR-Llama-8B | 11.9 | 8.5 | 72 |
| Trendyol-7B v1.8 | 12.5 | 11.3 | 50 |
| Mistral-7B-v0.3 | 13.0 | 5.9 | 46 |

- **Q4_K_M her modelde en fazla 1 puan Türkçe doğruluğa mal oluyor** ve modelin en hızlı ayarı.
- **En doğru ayar Gemma-2-9B Q4_K_M; kalite–hız sınırındaki diğer noktalar Aya-Expanse-8B ve Trendyol-7B
  Q4_K_M** (%76'da saniyede 9.0 Türkçe içerik token'ı; %50'de saniyede en fazla Türkçe içerik, 11.3).
- Ham hız içerik hızı değil: Mistral-7B token cinsinden en hızlı model (13.0 tok/s), ama 2.21×'lik
  vergisi onu Türkçe içerikte en yavaşlar arasına düşürüyor (5.9).
- CPU'da fp16 hızı bellek yerleşimine bağlı: Cosmos ve Mistral fp16 iki kez ölçüldü, 2.5 ve 2.7 tok/s
  çıktı; aynı mimarideki ikizleri (Llama, Trendyol) 4.1–4.2'de çalıştı ve dördü de Q8_0 ile Q4_K_M'de
  0.4 tok/s farkla aynı. Modelleri Q8_0 ya da Q4_K_M'de karşılaştırın.

### Girdi uzunluğu

Q4_K_M, 16 çekirdek. 256 / 1.024 / 4.096 token'da prompt okuma ve boş ya da 4.096 token'lık
context ile üretim:

| Model | Okuma 256 | Okuma 1k | Okuma 4k (tok/s) | Üretim, boş | Üretim, 4k context |
|---|---|---|---|---|---|
| Qwen2.5-7B | 65.7 | 64.7 | 58.0 | 11.4 | 8.3 |
| Gemma-2-9B | 50.6 | 48.5 | 41.9 | 8.8 | 5.6 |
| Llama-3.1-8B | 62.3 | 60.2 | 52.6 | 11.9 | 8.3 |
| Trendyol-7B v1.8 | 62.1 | 59.7 | 53.6 | 12.9 | 7.9 |

Okuma 256'dan 4.096 token'a çıkınca %12–17 yavaşlıyor; context'te 4.096 token olduğunda üretim
%27–39 yavaşlıyor. Aynı Türkçe metin 1.2–2.2 kat daha fazla token tuttuğu için Türkçe bir konuşma
bu yavaş bölgeye İngilizceden daha erken giriyor: tokenizer vergisi uzun context'te bir kez daha ödeniyor.

## Özet

Açık 7–9B modelleri kullanan Türkçe bir kullanıcı için: engine Türkçeyi token cinsinden İngilizce kadar
hızlı servis ediyor, ama bir GPU saatinin ne kadar Türkçe içerik ürettiğini (aynı GPU'da saniyede 57 ile
119 içerik token'ı, Türkçe kelime başına 2 kat maliyet farkı) ve yük altında kullanıcının ilk token'ı ne
kadar beklediğini tokenizer belirliyor; Q4_K_M'e kadar quantization yedi modelin hiçbirinde Türkçeye
İngilizceden fazla mal olmuyor; on dört modelde Türkçe doğruluk %48 ile %90 arasında değişiyor ve model
boyutunu izlemiyor (4B Gemma-3 ve Qwen3 birçok 8B modeli geçiyor); kalan kalite farkı base model'de
(Gemma-2-9B'de küçük, Llama-3.1-8B ve Mistral-7B'de büyük) ve kısmen çevrilmiş benchmark'lar yüzünden şişiyor.

Hatalar incelenirken bulunan veri kalitesi notları: Belebele-TR'de çeviri kokan dil, GSM8K-TR'de anlam
hataları, Global-MMLU-TR'de terim hataları.

## Kısıtlar

- GPU servis ölçümleri paylaşımlı tek bir V100'ün ~8 GB'lık bütçesinde yapıldı: 9B'ye kadar 4-bit modeller,
  en fazla 8 eşzamanlı istek (Gemma: 2). GPU'da fp16/Q8, 9B'den büyük modeller ve yüksek eşzamanlılık bütçeye
  sığmadı. Engine karşılaştırması yalnızca Mistral-7B'yi (başlayabilen tek AWQ modeli) ve eager modda vLLM'i kapsıyor.
- Tek bir GPU nesli (V100, SM70): daha yeni GPU'lar mutlak hızları ve muhtemelen engine sıralamasını
  değiştirir; tokenizer'ın içerik throughput'una etkisi GPU'ya bağlı değil.
- Kalite ölçümleri llama.cpp (`b12470`), greedy decoding ve konfigürasyon başına tek koşuyla yapıldı:
  quantization karşılaştırması CPU'da, on dört modellik tablo GPU'da. GGUF dosyaları orijinal ağırlıklardan
  aynı sürümle üretildi (Qwen ve yeni modeller: yayımlanmış GGUF dosyaları).
- 250 eşleştirilmiş soru yaklaşık ±4 puanlık %95 aralıkları veriyor; daha küçük bir quantization vergisi dışlanamaz.
- Okuma, bilgi ve matematikte çevrilmiş benchmark'lar kullanıldı; yukarıdaki errata'ya bakın.

Eval seti: [HaticeB/tr-llm-bench-eval](https://huggingface.co/datasets/HaticeB/tr-llm-bench-eval)
(özgün sorular, üçüncü taraf soruların ID'leri, çeviri errata'sı).

## Yeniden üretmek için

```bash
pip install -e ".[dev,tok]"
pytest -q
python -m trbench.tokenizer_tax.fertility
python -m trbench.eval.build
bash scripts/phase4_cpu.sh Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b
python -m trbench.eval.analysis --model qwen2.5-7b
# GPU (sunucuda): bash scripts/gpu_bench.sh configs/gpu_llamacpp.yaml, ardından
python -m trbench.eval.gpu_report
```

Her sonuç satırı nereden geldiğini kaydediyor: `git_commit` (ve `git_dirty`), eval setinin ve GGUF
dosyasının SHA-256'sı, engine build'i ve sonucu değiştiren her ayarın (max tokens, context, template,
engine flag'leri) hash'i olan `fp`. Tekrar koşu yalnızca aynı `fp`'ye sahip satırları atlıyor; böylece
bir ayar değiştiğinde eski bir ölçüm asla sessizce yeniden kullanılmıyor.
`python -m trbench.eval.audit_pairs`, her GSM8K TR/EN çiftinin aynı sayıları içerdiğini kontrol ediyor:
40'ın 36'sı birebir aynı; kalan 4'ü elle okundu ve farklı yazılmış doğru çiftler
(13:00 ve 1 PM, %25 ve 0.25, $1.00 ve 1, "ikişer" ve "2").

Benchmark paylaşımlı bir sunucuda çalıştı. `trbench.guard` diğer GPU'ları ve bir health endpoint'ini
(`TRBENCH_SERVICE_HEALTH`) izliyor ve bir şey etkilenirse koşuyu durduruyor. 30 sn'de bir kontrol ediyor
ve art arda 2 ihlalde durduruyor, yani yaklaşık bir dakika içinde tepki veriyor; guard sürecinin kendisi
ölürse `scripts/gpu_bench.sh` benchmark'ı da durduruyor.

## Lisans

Kod: MIT (`LICENSE` dosyasına bakın). Üçüncü taraf benchmark soruları kendi lisanslarını koruyor ve
yeniden dağıtılmıyor; veri seti kartında listeleniyor.
