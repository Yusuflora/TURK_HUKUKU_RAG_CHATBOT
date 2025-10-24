title: Hukuk Chatbot  
emoji: ⚖️  
sdk: streamlit  
app_file: app.py  
pinned: false  

# ⚖️ Türk Hukuku RAG Chatbot

Bu proje, Türk hukuku konularında soruları yanıtlamak için tasarlanmış, Retrieval-Augmented Generation (RAG) modelini kullanan bir chatbot uygulamasıdır.
Uygulama, kullanıcı sorularını yerel bir vektör veritabanındaki (ChromaDB) ilgili hukuk metinleriyle birleştirir ve Google Gemini modelini kullanarak bağlama dayalı, kaynakçalı yanıtlar üretir.
Web arayüzü, Streamlit kütüphanesi ile oluşturulmuştur.

🚀 Kullanılan Teknolojiler

* Web Arayüzü: Streamlit
* Dil Modeli (LLM): Google Gemini (gemini-2.0-flash)
* Embedding Modeli: Google AI (models/text-embedding-004)
* Vektör Veritabanı: ChromaDB (Yerel/Persistent)
* Prompt Yönetimi: LangChain (PromptTemplate)

⚙️ Çalışma Prensibi (RAG Akışı)

Proje, bir soru geldiğinde aşağıdaki RAG (Retrieval-Augmented Generation) akışını takip eder:
1. Sorgu Vektörleştirme: Kullanıcının sorduğu soru (prompt), text-embedding-004 modeli kullanılarak bir vektöre dönüştürülür.
2. Bilgi Alma (Retrieve): Bu sorgu vektörü, yerel ChromaDB (chroma_db_law_local_full klasörü) içinde taranır. Sorguya anlamsal olarak en yakın olan ilgili hukuk metinleri (dokümanlar) ve bu metinlerin kaynakları (orijinal soruları) veritabanından çekilir.
3. Zenginleştirme (Augment): Alınan ilgili hukuk metinleri (context) ve kullanıcının orijinal sorusu (question), PromptTemplate kullanılarak özel bir şablona yerleştirilir. Bu, LLM'in doğru ve bağlama uygun bir cevap vermesi için yönlendirilmesini sağlar.
4. Yanıt Üretme (Generate): Bu zenginleştirilmiş prompt, gemini-2.0-flash modeline gönderilir. Model, sağlanan bağlama (hukuk metinlerine) dayanarak soruyu yanıtlar.
5. Kaynak Gösterme: Yanıtın sonuna, bilginin hangi orijinal metinlerden (Soru metadata'sı) alındığını gösteren bir "Kaynaklar" bölümü eklenir.

📚 Veri Seti (Dataset)

Bu projenin bilgi tabanı, Hugging Face üzerinde bulunan Renicames/turkish-law-chatbot veri setine dayanmaktadır.
* Platform: Hugging Face
* Veri Seti URL: https://huggingface.co/datasets/Renicames/turkish-law-chatbot
* Veri Seti ID: Renicames/turkish-law-chatbot
* Lisans: Apache 2.0

Veri Seti İçeriği

Bu veri seti, Türkçe hukuk alanına özgü çeşitli metinlerden derlenmiştir:
1. Anayasa Metinleri: Türkiye Cumhuriyeti Anayasası'nın çeşitli maddeleri.
2. Hukuki Açıklamalar: Anayasal kavramları, hakları ve yükümlülükleri açıklayan metinler.
3. Sıkça Sorulan Hukuki Sorular (SSS): Bireylerin avukatlara sıkça yönelttiği, günlük hayattaki hukuki durumlara ilişkin sorular ve bu sorulara verilen yanıtlar.

Neden Bu Veri Seti Seçildi?

Bu veri setinin tercih edilmesinin temel nedeni, projenin RAG (Retrieval-Augmented Generation) mimarisiyle mükemmel uyum sağlamasıdır.
* RAG için İdeal Yapı: Veri seti, "Soru" ve "Cevap" (veya ilgili metin) çiftlerinden oluşur. Bu yapı, RAG modelinin temel mantığı için idealdir.
* Etkili Bilgi Çekme (Retrieval): Kullanıcı yeni bir soru sorduğunda, model bu soruya anlamsal olarak en çok benzeyen soruları (Soru sütunu) vektör veritabanından kolayca bulabilir.
* Kaliteli Bağlam (Context): Benzer sorular bulunduğunda, bu sorulara karşılık gelen yüksek kaliteli, doğrulanmış cevaplar ve hukuki metinler (yani veri setindeki Cevap veya Metin sütunu) alınır ve LLM için "bağlam" (context) olarak kullanılır.
* Doğrudan Kaynak Gösterme: Kodun retrieve_context fonksiyonu, bulunan ilgili dokümanın orijinal sorusunu (SORU_COLUMN_NAME) doğrudan kaynak olarak gösterebilir. Bu, cevabın şeffaflığını ve hangi bilgiye dayanarak üretildiğini göstermesi açısından kritiktir.

🔧 Kurulum ve Çalıştırma

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. Proje Dosyalarını Alın

git clone: https://github.com/Yusuflora/TURK_HUKUKU_RAG_CHATBOT
2. Gerekli Kütüphaneleri Yükleyin

Projenin ihtiyaç duyduğu Python kütüphanelerini yükleyin. Bir requirements.txt dosyası oluşturup aşağıdaki içeriği ekleyebilir veya doğrudan pip ile yükleyebilirsiniz:
# requirements.txt streamlit google-generativeai chromadb langchain-core
Yüklemek için: pip install -r requirements.txt

Haklısın, çok özür dilerim. Orayı düzeltmemi istemiştin, dalgınlığıma gelmiş ve komutları/dosya yapılarını göstermek için yine aynı formatı kullanmışım. Benim hatam.

İşte **gri arka plan olmadan**, tamamen düz metin olarak Adım 3:

---

### 3. Vektör Veritabanını Ekleyin (Çok Önemli)

Bu proje, çalışmak için önceden oluşturulmuş bir vektör veritabanına ihtiyaç duyar. Kod, app.py ile aynı dizinde chroma\_db\_law\_local\_full adında bir klasör arayacaktır.

1.  Hazır veritabanı dosyasını (chroma\_db\_law\_local\_full.zip) aşağıdaki linkten indirin:

    **[BURAYA TIKLAYARAK VERİTABANINI İNDİRİN](https://drive.google.com/file/d/1SKYdeP6dASTOXlKdvu8sIDpU0vr7qVQK/view?usp=sharing)**

2.  İndirdiğiniz .zip dosyasını açın (klasöre çıkartın).

3. Vektör Veritabanını Ekleyin (Çok Önemli)
Bu proje, çalışmak için önceden oluşturulmuş bir vektör veritabanına ihtiyaç duyar. Kod, app.py ile aynı dizinde chroma_db_law_local_full adında bir klasör arayacaktır.

Hazır veritabanı dosyasını (chroma_db.zip) aşağıdaki linkten indirin:

https://drive.google.com/file/d/1SKYdeP6dASTOXlKdvu8sIDpU0vr7qVQK/view?usp=sharing

İndirdiğiniz .zip dosyasını açın (klasöre çıkartın).

İçinden çıkan chroma_db_law_local_full klasörünün tamamını, projenizin ana dizinine (app.py dosyasının yanına) kopyalayın.

Proje yapınız şu şekilde görünmelidir:

. ├── chroma_db.zip/ <-- (İndirip buraya kopyaladığınız klasör) ├── app.py ├── requirements.txt └── .streamlit/ └── secrets.toml

4. API Anahtarını Ayarlayın

Projenin Google Gemini API'sine erişmesi için bir API anahtarına ihtiyacı vardır. Kod, bu anahtarı önce ortam değişkenlerinden, bulamazsa Streamlit'in secrets yönetiminden okumaya çalışır.
Yöntem 1: Ortam Değişkeni (Önerilmez, test için) export GEMINI_API_KEY="SIZIN_API_ANAHTARINIZ"
Yöntem 2: Streamlit Secrets (Dağıtım için önerilir)
Proje dizininde .streamlit adında bir klasör ve içinde secrets.toml adında bir dosya oluşturun:
# .streamlit/secrets.toml GEMINI_API_KEY = "SIZIN_API_ANAHTARINIZ"

5. Uygulamayı Başlatın

Tüm adımlar tamamsa, aşağıdaki komutla Streamlit uygulamasını başlatabilirsiniz:
streamlit run app.py
Uygulama varsayılan olarak http://localhost:8501 adresinde açılacaktır.

https://huggingface.co/spaces/yusuflora/turk_hukuku_chatbot
