# ===============================================
#     *** TÜRK HUKUKU RAG CHATBOT - APP.PY ***
# ===============================================
# Bu dosya, Streamlit kütüphanesini kullanarak
# RAG modelimiz için bir web arayüzü sağlar.
# ===============================================

### DEBUG ###
print("--- APP.PY BAŞLADI ---")

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Gemini Güvenlik Ayarları için
import chromadb
from langchain_core.prompts import PromptTemplate
import time
import os
import shutil # Dosya işlemleri için

### DEBUG ###
print("--- Importlar tamamlandı ---")

# --- Konfigürasyon ve Global Değişkenler ---

### DEBUG ###
print("--- Konfigürasyon başlıyor ---")

# API Anahtarını al: Önce ortam değişkenlerinden (os.environ) dener,
# bulamazsa Streamlit'in kendi 'secrets' (st.secrets) yönetiminden dener.
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        ### DEBUG ###
        print("--- API Key secrets'ten alındı ---")
    except Exception:
        ### DEBUG ###
        print("--- HATA: API Key bulunamadı ---")
        st.error("Gemini API Anahtarı 'GEMINI_API_KEY' bulunamadı! Lütfen Hugging Face Space Secrets'e ekleyin.")
        st.stop() # Anahtar yoksa uygulama durur.
else:
    ### DEBUG ###
    print("--- API Key ortam değişkeninden alındı ---")


# Proje Sabitleri
DB_PATH = "./chroma_db_law_local_full"
COLLECTION_NAME = "hukuk_tr_collection_full_local"
MODEL_NAME_LLM = "gemini-2.0-flash"
MODEL_NAME_EMBEDDING = "models/text-embedding-004"
SORU_COLUMN_NAME = "Soru"
DATASET_ID_STR = "Renicames/turkish-law-chatbot"

# Gemini API'yi yapılandır
try:
    genai.configure(api_key=API_KEY)
    ### DEBUG ###
    print("--- Gemini API yapılandırıldı ---")
except Exception as config_e:
    ### DEBUG ###
    print(f"--- HATA: Gemini API yapılandırılamadı: {config_e} ---")
    st.error(f"API yapılandırılamadı: {config_e}")
    st.stop()

# --- Yardımcı Fonksiyonlar ---

### DEBUG ###
print("--- Yardımcı fonksiyonlar tanımlanıyor ---")

# Embedding Fonksiyonu
def embed_content_with_retry(content, model=MODEL_NAME_EMBEDDING, task_type="RETRIEVAL_DOCUMENT", max_retries=3, initial_delay=1):
    ### DEBUG ###
    # print(f"--- embed_content_with_retry çağrıldı (tip: {task_type}) ---") # Çok fazla log üretebilir, şimdilik kapalı
    delay = initial_delay; last_exception = None; is_batch = isinstance(content, list)
    for attempt in range(max_retries):
        try:
            current_task_type = task_type if isinstance(content, str) and task_type == 'RETRIEVAL_QUERY' else 'RETRIEVAL_DOCUMENT'
            result = genai.embed_content(model=model, content=content, task_type=current_task_type)
            embedding_key = 'embedding' if 'embedding' in result else ('embeddings' if 'embeddings' in result else None)
            if embedding_key: return result[embedding_key]
            else: raise ValueError("Embedding sonucu anahtar içermiyor.")
        except Exception as e:
            ### DEBUG ###
            print(f"--- HATA: Embedding deneme {attempt + 1}: {e} ---")
            st.toast(f"Embedding hatası (Deneme {attempt + 1}): {e}", icon="⚠️")
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay); delay *= 2 # Hata sonrası bekleme (exponential backoff)
            else:
                if is_batch: return [None] * len(content)
                else: raise last_exception
    return None

# Chroma Vektör Veritabanını Yükleme Fonksiyonu
@st.cache_resource
def get_chroma_collection():
    ### DEBUG ###
    print("--- get_chroma_collection fonksiyonu başladı ---")
    try:
        ### DEBUG ###
        print(f"--- DB Yolu kontrol ediliyor: {DB_PATH} ---")
        # DB_PATH klasörünün var olup olmadığını kontrol et
        if not os.path.exists(DB_PATH):
            ### DEBUG ###
            print(f"--- HATA: DB yolu bulunamadı: {DB_PATH} ---")
            st.error(f"Veritabanı yolu bulunamadı: '{DB_PATH}'.")
            st.error("Lütfen Colab'de oluşturduğunuz 'chroma_db_law_local_full' klasörünü bu reponun ana dizinine yüklediğinizden emin olun.")
            return None
        
        effective_db_path = DB_PATH 
        ### DEBUG ###
        print(f"--- Kullanılacak DB yolu: {effective_db_path} ---")

        # 'PersistentClient', veritabanının diskten (belirtilen yoldan)
        # kalıcı olarak yüklenmesini sağlar.
        ### DEBUG ###
        print("--- chromadb.PersistentClient oluşturuluyor... ---")
        client = chromadb.PersistentClient(path=effective_db_path)
        ### DEBUG ###
        print("--- PersistentClient oluşturuldu. ---")
        
        # Koleksiyonu isimle çağır (oluşturmaya çalışma, SADECE yükle)
        ### DEBUG ###
        print(f"--- Koleksiyon '{COLLECTION_NAME}' alınıyor... ---")
        collection = client.get_collection(name=COLLECTION_NAME)
        ### DEBUG ###
        print(f"--- Koleksiyon alındı. Öğe sayısı: {collection.count()} ---")
        
        st.info(f"Yerel Chroma koleksiyonu '{COLLECTION_NAME}' yüklendi (Öğe Sayısı: {collection.count()}).")
        
        # DB'nin boş olup olmadığını kontrol et
        if collection.count() == 0:
            ### DEBUG ###
            print("--- UYARI: Chroma koleksiyonu boş! ---")
            st.error("⚠️ Yerel Chroma koleksiyonu boş! Veritabanı dosyaları bozuk veya yanlış yüklenmiş olabilir.")
        
        ### DEBUG ###
        print("--- get_chroma_collection fonksiyonu başarıyla tamamlandı ---")
        return collection
        
    except Exception as e:
        ### DEBUG ###
        print(f"--- HATA: Chroma yüklenirken: {e} ---")
        st.error(f"Yerel Chroma yüklenirken/alınırken hata: {e}")
        st.error(f"Veritabanı dosyalarının ('{DB_PATH}' klasörü) 'app.py' ile aynı dizinde olduğundan emin olun.")
        return None

# Retriever Fonksiyonu (Bilgi Çekici)
def retrieve_context(query: str, collection, k: int = 3):
    ### DEBUG ###
    # print(f"--- retrieve_context çağrıldı: '{query[:50]}...' ---") # Çok fazla log üretebilir
    if collection is None:
        ### DEBUG ###
        print("--- retrieve_context: Koleksiyon None, çıkılıyor. ---")
        return "Veritabanı bağlantısı kurulamadı.", ""
    try:
        query_embedding = embed_content_with_retry(query, task_type='RETRIEVAL_QUERY')
        if not query_embedding:
            ### DEBUG ###
            print("--- retrieve_context: Sorgu embed edilemedi. ---")
            return "Sorgu vektöre çevrilirken hata oluştu.", ""

        results = collection.query(query_embeddings=[query_embedding], n_results=k, include=['documents', 'metadatas'])

        if results and results.get('documents') and results['documents'][0]:
            context_list = []; sources = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source_info = metadata.get(SORU_COLUMN_NAME, f"ID: {metadata.get('source_id', 'Bilinmiyor')}")
                context_list.append(f"Metin: {doc}")
                sources.append(source_info)

            context_str = "\n\n".join(context_list)
            source_str = "\n".join([f"- {s}" for s in set(sources)])
            ### DEBUG ###
            # print(f"--- retrieve_context: {len(sources)} kaynak bulundu. ---")
            return context_str, source_str
        else:
            ### DEBUG ###
            print("--- retrieve_context: İlgili bilgi bulunamadı. ---")
            return "İlgili bilgi bulunamadı.", ""
    except Exception as e:
        ### DEBUG ###
        print(f"--- HATA: Retriever: {e} ---")
        st.error(f"Retriever hatası: {e}");
        return f"Bilgi alınırken hata oluştu.", ""

# Prompt Şablonu (RAG için)
template_str_app = '''Sen Türk Hukuku alanında uzman bir yapay zeka asistanısın...
Bağlam:
{context}
Soru:
{question}
Yanıt:
'''
prompt_template_lc = PromptTemplate.from_template(template_str_app)

### DEBUG ###
print("--- Prompt şablonu oluşturuldu ---")

# --- Streamlit Arayüzü Başlangıcı ---
st.set_page_config(page_title="🇹🇷 Türk Hukuku RAG Chatbot", page_icon="⚖")
st.title("⚖ Türk Hukuku RAG Chatbot")
st.caption(f"Veri Seti: {DATASET_ID_STR} (HF) | Vektör DB: Yerel Chroma | Model: {MODEL_NAME_LLM}")

### DEBUG ###
print("--- Streamlit sayfa konfigürasyonu yapıldı ---")

# --- Gemini Güvenlik Ayarları (Safety Settings) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
# --------------------------------------------------------

### DEBUG ###
print("--- Güvenlik ayarları tanımlandı ---")

# LLM (Dil Modeli) Yükleme Fonksiyonu
@st.cache_resource
def initialize_llm():
    ### DEBUG ###
    print("--- initialize_llm fonksiyonu başladı ---")
    try:
        ### DEBUG ###
        print(f"--- GenerativeModel ({MODEL_NAME_LLM}) oluşturuluyor... ---")
        llm_model = genai.GenerativeModel(
            model_name=MODEL_NAME_LLM,
            generation_config={"temperature": 0.3}, # Yaratıcılığı düşük tut
            safety_settings=safety_settings
            )
        ### DEBUG ###
        print("--- GenerativeModel oluşturuldu. ---")
        st.success(f"Gemini modeli ({MODEL_NAME_LLM}) başarıyla yüklendi (Güvenlik: BLOCK_ONLY_HIGH).")
        ### DEBUG ###
        print("--- initialize_llm fonksiyonu başarıyla tamamlandı ---")
        return llm_model
    except Exception as e:
        ### DEBUG ###
        print(f"--- HATA: LLM yüklenirken: {e} ---")
        st.error(f"LLM yüklenirken hata oluştu: {e}"); return None

# Ana bileşenleri (LLM ve DB) yükle
### DEBUG ###
print("--- Ana bileşenler yükleniyor (LLM)... ---")
llm_model = initialize_llm()
### DEBUG ###
print(f"--- LLM yüklemesi tamamlandı. Sonuç: {'Başarılı' if llm_model else 'Başarısız'} ---")

### DEBUG ###
print("--- Ana bileşenler yükleniyor (Chroma DB)... ---")
chroma_collection = get_chroma_collection()
### DEBUG ###
print(f"--- Chroma DB yüklemesi tamamlandı. Sonuç: {'Başarılı' if chroma_collection else 'Başarısız'} ---")


if llm_model is None or chroma_collection is None:
    ### DEBUG ###
    print("--- HATA: LLM veya Chroma yüklenemedi, uygulama durduruluyor. ---")
    st.error("Ana kaynaklar (LLM veya Vektör DB) yüklenemedi. Uygulama durduruluyor.")
    st.stop()

### DEBUG ###
print("--- Ana bileşenler başarıyla yüklendi ---")

# RAG Cevap Fonksiyonu (Ana Mantık)
def get_response_from_rag(user_query):
    ### DEBUG ###
    print(f"--- get_response_from_rag çağrıldı: '{user_query[:50]}...' ---")
    try:
        # 1. Bilgiyi Çek (Retrieve)
        ### DEBUG ###
        print("--- Retriever çağrılıyor... ---")
        retrieved_context, sources_str = retrieve_context(user_query, chroma_collection)
        ### DEBUG ###
        print(f"--- Retriever döndü. Context var mı?: {bool(retrieved_context and 'bulunamadı' not in retrieved_context)}, Kaynak var mı?: {bool(sources_str)} ---")

        # 2. Prompt'u Hazırla (Augment)
        formatted_prompt = prompt_template_lc.format(question=user_query, context=retrieved_context)
        ### DEBUG ###
        # print(f"--- Formatlanmış Prompt (ilk 100 char): {formatted_prompt[:100]}... ---") # Çok uzun olabilir

        # 3. Cevap Üret (Generate)
        ### DEBUG ###
        print("--- LLM generate_content çağrılıyor... ---")
        response = llm_model.generate_content(formatted_prompt)
        ### DEBUG ###
        print("--- LLM generate_content döndü. ---")

        try:
            answer = response.text
            ### DEBUG ###
            print(f"--- LLM Cevabı (ilk 50 char): {answer[:50]}... ---")
            
            # Güvenlik Engeli Kontrolü
            if not answer and response.prompt_feedback.block_reason:
                ### DEBUG ###
                print(f"--- UYARI: Yanıt güvenlik nedeniyle engellendi: {response.prompt_feedback.block_reason} ---")
                st.warning(f"⚠️ Yanıt güvenlik nedeniyle engellendi: {response.prompt_feedback.block_reason}")
                return "Üzgünüm, ürettiğim yanıt güvenlik politikalarımız nedeniyle engellendi."
            
            # 4. Kaynakları Cevaba Ekle
            if sources_str and "bulunamadı" not in answer:
                answer += f"\n\n---\n*Kaynaklar (İlgili Orijinal Sorular):*\n{sources_str}"
                ### DEBUG ###
                print("--- Kaynaklar cevaba eklendi. ---")
            return answer

        except Exception as resp_e:
            ### DEBUG ###
            print(f"--- HATA: Yanıt işlenirken: {resp_e} ---")
            st.error(f"Yanıt (response) işlenirken hata: {resp_e}")
            st.warning(f"Modelin ham cevabı: {response}")
            return "Yanıt alınırken bir hata oluştu."

    except Exception as e:
        ### DEBUG ###
        print(f"--- HATA: RAG süreci genel hata: {e} ---")
        st.error(f"RAG süreci hatası: {e}")
        return f"Üzgünüm, cevap üretilirken genel bir hata oluştu."

# --- Chat Arayüzü Mantığı ---

### DEBUG ###
print("--- Chat arayüzü mantığı başlıyor ---")

# Chat geçmişini Streamlit'in 'session_state' hafızasında tut
if "messages" not in st.session_state:
    st.session_state.messages = [{ "role": "assistant", "content": "Merhaba! Türk hukuku hakkında ne öğrenmek istersiniz?" }] # Başlangıç mesajı

# Geçmişteki mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni giriş (prompt) al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # 1. Kullanıcının mesajını geçmişe ve ekrana ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistanın cevabını hazırla
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Cevap gelene kadar boş bir alan ayır
        with st.spinner("Düşünüyor... (Veritabanı taranıyor ve cevap üretiliyor)"):
            ### DEBUG ###
            print("--- Spinner aktif, RAG fonksiyonu çağrılıyor ---")
            assistant_response = get_response_from_rag(prompt)
            ### DEBUG ###
            print("--- RAG fonksiyonu döndü, spinner kapanıyor ---")
        
        # 3. Asistanın cevabını ekrana ve geçmişe ekle
        message_placeholder.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

### DEBUG ###
print("--- APP.PY SONLANDI (Streamlit devralıyor) ---")
