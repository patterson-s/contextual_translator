import streamlit as st
import cohere
import os
from typing import List
import time

st.set_page_config(
    page_title="Traducteur Multilingue",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def init_cohere_client():
    """Initialise le client Cohere en utilisant les secrets/variables d'environnement"""
    api_key = None
    
    # Essayer les secrets Streamlit d'abord (pour le déploiement hébergé)
    try:
        api_key = st.secrets["COHERE_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Repli sur la variable d'environnement (pour le développement local)
        api_key = os.getenv('COHERE_API_KEY')
    
    if not api_key:
        st.error("Clé API Cohere introuvable. Veuillez configurer COHERE_API_KEY dans les secrets Streamlit ou les variables d'environnement.")
        st.stop()
    
    return cohere.ClientV2(api_key)

# Configuration du modèle pour Aya 32B
MODEL_CONFIG = {
    "name": "Aya Expanse 32B", 
    "description": "Modèle multilingue 32B avec contexte étendu",
    "context_length": 131072,
    "max_output_tokens": 4096,
    "recommended_chunk_size": 2000
}

MODEL_NAME = "c4ai-aya-expanse-32b"

AYA_LANGUAGES = [
    "Arabe", "Chinois (Simplifié)", "Chinois (Traditionnel)", "Tchèque", "Néerlandais", 
    "Anglais", "Français", "Allemand", "Grec", "Hébreu", "Hindi", "Indonésien", 
    "Italien", "Japonais", "Coréen", "Persan", "Polonais", "Portugais", 
    "Roumain", "Russe", "Espagnol", "Turc", "Ukrainien", "Vietnamien"
]

def split_text_into_chunks(text: str, chunk_size: int = 2000, overlap: int = 50) -> List[str]:
    """Divise le texte en segments avec chevauchement pour maintenir le contexte"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        chunk = text[start:end]
        
        # Essayer de couper aux limites de phrases
        last_sentence = max(
            chunk.rfind('.'), 
            chunk.rfind('!'), 
            chunk.rfind('?')
        )
        
        if last_sentence > chunk_size * 0.7:
            end = start + last_sentence + 1
        else:
            # Repli sur les limites de mots
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.5:
                end = start + last_space
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def translate_chunk(client, text: str, target_lang: str, source_lang: str, temperature: float, context: str) -> str:
    """Traduit un seul segment en utilisant Cohere Aya"""
    system_prompt = f"{context.strip()} Traduisez le texte {source_lang} suivant en {target_lang}:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=MODEL_CONFIG["max_output_tokens"]
        )
        return "".join(item.text for item in response.message.content if hasattr(item, 'text'))
    except Exception as e:
        st.error(f"Échec de la traduction pour le segment: {str(e)}")
        return f"[Échec de la traduction: {str(e)}]"

def main():
    st.title("🌍 Traducteur Multilingue")
    st.markdown("Traduisez du texte avec une prise en compte contextuelle")
    
    # Initialiser le client Cohere
    client = init_cohere_client()
    
    # Configuration de la barre latérale
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        # Sélection des langues
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox(
                "De",
                AYA_LANGUAGES,
                index=AYA_LANGUAGES.index("Français")
            )
        
        with col2:
            target_lang = st.selectbox(
                "Vers",
                AYA_LANGUAGES,
                index=AYA_LANGUAGES.index("Anglais")
            )
        
        st.divider()
        
        # Configuration des segments
        with st.expander("📄 Paramètres Avancés"):
            default_chunk_size = MODEL_CONFIG["recommended_chunk_size"]
            max_chunk_size = min(3000, MODEL_CONFIG["context_length"] // 4)
            
            chunk_size = st.slider(
                "Taille des segments (caractères)",
                min_value=500,
                max_value=max_chunk_size,
                value=default_chunk_size,
                step=100
            )
            
            chunk_overlap = st.slider(
                "Chevauchement des segments (caractères)",
                min_value=0,
                max_value=min(300, chunk_size // 4),
                value=50,
                step=10
            )
            
            temperature = st.slider(
                "Température",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1
            )
        
        # Informations sur le modèle
        with st.expander("ℹ️ À propos"):
            st.write(f"**Modèle:** {MODEL_CONFIG['name']}")
            st.write(f"**Description:** {MODEL_CONFIG['description']}")
            st.write(f"**Longueur de contexte:** {MODEL_CONFIG['context_length']:,} tokens")
            st.write(f"**Langues:** 23 supportées")
            
            st.markdown("---")
            st.markdown("**Propulsé par Cohere AI**")
            st.markdown("Les modèles Aya Expanse de Cohere AI utilisent des techniques d'entraînement avancées pour l'excellence multilingue sur 23 langues.")
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"🎯 Contexte")
        context_text = st.text_area(
            "Contexte de traduction:",
            value="Vous êtes un traducteur qui m'aide à traduire des messages pour mon superviseur. Mon superviseur et moi parlons de manière polie mais informelle. Je veux être respectueux mais pas guindé. Répondez uniquement avec la traduction et rien d'autre. Ne pas inclure de préambule, d'explication ou de commentaire. Donnez seulement le texte traduit. Voici le texte à traduire:",
            height=120
        )
        
        st.subheader(f"📝 Texte ({source_lang})")
        input_text = st.text_area(
            "Texte à traduire:",
            height=250,
            placeholder=f"Entrez votre texte en {source_lang} ici...",
            key="input_text_area"
        )
        
        # Option de téléchargement de fichier
        uploaded_file = st.file_uploader(
            "Télécharger un fichier texte",
            type=['txt']
        )
        
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode('utf-8')
            st.success(f"Fichier téléchargé ! ({len(input_text):,} caractères)")
        
        # Comptage des caractères et segments estimés
        if input_text:
            char_count = len(input_text)
            estimated_chunks = max(1, (char_count + chunk_size - 1) // chunk_size)
            st.info(f"📊 {char_count:,} caractères • ~{estimated_chunks} segments")
    
    with col2:
        st.subheader(f"🌐 Traduction ({target_lang})")
        
        # Create a form to enable Enter key functionality
        with st.form(key="translation_form", clear_on_submit=False):
            translate_clicked = st.form_submit_button("Traduire", type="primary", use_container_width=True)
        
        if translate_clicked:
            if not input_text.strip():
                st.warning("Veuillez entrer du texte à traduire")
            elif not context_text.strip():
                st.warning("Veuillez fournir un contexte de traduction")
            elif source_lang == target_lang:
                st.warning("Les langues source et cible ne peuvent pas être identiques")
            else:
                # Afficher le progrès de la traduction
                with st.spinner(f"Traduction avec {MODEL_CONFIG['name']}..."):
                    start_time = time.time()
                    
                    # Diviser le texte en segments
                    chunks = split_text_into_chunks(input_text, chunk_size, chunk_overlap)
                    
                    # Suivi du progrès
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    translated_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            status_text.text(f"Traduction du segment {i+1} sur {len(chunks)} avec {MODEL_CONFIG['name']}...")
                            
                            translated_chunk = translate_chunk(
                                client, chunk, target_lang, source_lang, temperature, context_text
                            )
                            translated_chunks.append(translated_chunk)
                            
                            # Mettre à jour le progrès
                            progress_bar.progress((i + 1) / len(chunks))
                            time.sleep(0.1)
                    
                    # Combiner les segments traduits
                    full_translation = " ".join(translated_chunks)
                    translation_time = time.time() - start_time
                    
                    # Effacer les indicateurs de progrès
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Stocker la traduction dans l'état de session
                    st.session_state.translation = full_translation
                    st.session_state.chunks_processed = len(translated_chunks)
                    st.session_state.translation_time = translation_time
                    st.session_state.model_used = MODEL_CONFIG['name']
        
        # Afficher les résultats de traduction
        if 'translation' in st.session_state:
            st.text_area(
                "Traduction:",
                value=st.session_state.translation,
                height=400,
                help=f"Traité {st.session_state.chunks_processed} segments en {st.session_state.translation_time:.1f}s avec {st.session_state.model_used}"
            )
            
            # Bouton de téléchargement
            st.download_button(
                label="Télécharger",
                data=st.session_state.translation,
                file_name=f"traduction_{source_lang}_vers_{target_lang}.txt",
                mime="text/plain"
            )
            
            # Statistiques
            with st.expander("📊 Statistiques de Traduction"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Caractères d'entrée", f"{len(input_text):,}")
                with col2:
                    st.metric("Caractères de sortie", f"{len(st.session_state.translation):,}")
                with col3:
                    st.metric("Segments traités", st.session_state.chunks_processed)
                with col4:
                    st.metric("Temps de traduction", f"{st.session_state.translation_time:.1f}s")

    # Pied de page avec informations sur le modèle
    with st.expander("💡 Informations sur le Modèle"):
        st.markdown(f"""
        **Modèle Actuel:** {MODEL_CONFIG['name']}
        
        **Langues Supportées:** {', '.join(AYA_LANGUAGES)}
        
        **À propos d'Aya Expanse:**
        - Modèles multilingues conçus pour une traduction de haute qualité
        - Optimisés pour 23 langues avec des techniques d'entraînement avancées
        - Traduction contextuelle avec support de segmentation pour les textes longs
        """)

if __name__ == "__main__":
    main()