import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import numpy as np

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Set page config with improved theme
st.set_page_config(
    page_title="Aural Prism Song Recommendation Engine",
    layout="wide",
    page_icon="🎵"
)

# Define all CSS with updated dark color scheme
st.markdown("""
<style>
    /* Main container styling - dark sophisticated theme */
    .stApp {
        background: linear-gradient(135deg, #121212 0%, #000000 100%);
        color: #f0f0f0;
    }
    
    /* Card styling */
    .card {
        background-color: rgba(30, 30, 30, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(80, 80, 80, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #8466b5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #9b7dd1;
        transform: scale(1.05);
    }
    
    /* Slider styling */
    .stSlider>div>div>div>div {
        background-color: #8466b5;
    }
    
    /* Radio button styling */
    .stRadio>div {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 8px;
        padding: 10px;
    }
    
    .stRadio>div>div>div>label {
        color: #f0f0f0;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px;
        transition: all 0.3s;
    }
    
    .stRadio>div>div>div>label:hover {
        background-color: rgba(132, 102, 181, 0.3);
    }
    
    /* Text input styling */
    .stTextArea>div>div>textarea {
        background-color: rgba(30, 30, 30, 0.7);
        color: #f0f0f0;
        border-radius: 8px;
        border: 1px solid rgba(80, 80, 80, 0.4);
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0 !important;
    }
    
    /* Custom spotify iframe container */
    .spotify-iframe {
        background-color: #282828;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        transition: transform 0.3s;
        border: 1px solid rgba(80, 80, 80, 0.3);
    }
    
    .spotify-iframe:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Custom tabs */
    .stTabs>div>div>div>div {
        color: #f0f0f0;
    }
    
    /* Custom expander */
    .stExpander>div>div {
        background-color: rgba(30, 30, 30, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(80, 80, 80, 0.2);
    }
    
    /* Custom select box */
    .stSelectbox>div>div>div>div {
        color: #f0f0f0;
        background-color: rgba(30, 30, 30, 0.7);
    }
    
    /* Success messages */
    .stAlert {
        border-radius: 8px;
        background-color: rgba(50, 50, 50, 0.9);
    }
    
    /* Accent color adjustments */
    .accent-text {
        color: #9b7dd1;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 1px solid rgba(132, 102, 181, 0.5);
        margin: 20px 0;
    }
    
    /* Success message styling */
    .element-container:has(.stSuccess) > div {
        background-color: rgba(40, 40, 40, 0.9) !important;
        border: 1px solid rgba(132, 102, 181, 0.5) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    df = pd.read_csv("filtered_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

# Text-based emotion detection function
def detect_emotion_from_text(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.5:
        return 'Happy'
    elif scores['compound'] <= -0.5:
        return 'Sad'
    elif scores['compound'] > 0 and scores['compound'] < 0.5:
        return 'Surprise' if scores['pos'] > 0.3 else 'Neutral'
    elif scores['compound'] < 0 and scores['compound'] > -0.5:
        return 'Angry' if scores['neg'] > 0.3 else 'Fear'
    else:
        return 'Neutral'

# Advanced emotion detection
def advanced_emotion_detection(text):
    emotion_keywords = {
        'Happy': ['happy', 'joy', 'delighted', 'excited', 'cheerful', 'pleased', 'thrilled', 'elated', 'jubilant', 'love', 'glad'],
        'Sad': ['sad', 'unhappy', 'depressed', 'gloomy', 'miserable', 'heartbroken', 'melancholy', 'down', 'blue', 'grief'],
        'Angry': ['angry', 'mad', 'furious', 'outraged', 'annoyed', 'irritated', 'enraged', 'infuriated', 'frustrated'],
        'Fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'nervous', 'worried', 'panic', 'dread', 'horror'],
        'Surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned', 'unexpected', 'wow'],
        'Disgust': ['disgusted', 'repulsed', 'revolted', 'sickened', 'appalled', 'gross', 'yuck', 'eww'],
        'Neutral': ['okay', 'fine', 'neutral', 'normal', 'balanced', 'indifferent', 'neither', 'meh']
    }
    
    text = text.lower()
    emotion_counts = {emotion: 0 for emotion in emotion_keywords}
    
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', text))
            emotion_counts[emotion] += count
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    if max(emotion_counts.values()) > 0:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        return dominant_emotion
    else:
        if scores['compound'] >= 0.5:
            return 'Happy'
        elif scores['compound'] <= -0.5:
            return 'Sad'
        elif scores['compound'] > 0 and scores['compound'] < 0.5:
            return 'Surprise' if scores['pos'] > 0.3 else 'Neutral'
        elif scores['compound'] < 0 and scores['compound'] > -0.5:
            return 'Angry' if scores['neg'] > 0.3 else 'Fear'
        else:
            return 'Neutral'

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

emotion_presets = {
    'Angry': {'acousticness': 0.2, 'danceability': 0.6, 'energy': 0.8, 'instrumentalness': 0.1, 'valence': 0.3, 'tempo': 140.0},
    'Disgust': {'acousticness': 0.5, 'danceability': 0.3, 'energy': 0.4, 'instrumentalness': 0.2, 'valence': 0.2, 'tempo': 100.0},
    'Fear': {'acousticness': 0.7, 'danceability': 0.3, 'energy': 0.2, 'instrumentalness': 0.5, 'valence': 0.1, 'tempo': 90.0},
    'Happy': {'acousticness': 0.4, 'danceability': 0.8, 'energy': 0.7, 'instrumentalness': 0.1, 'valence': 0.9, 'tempo': 120.0},
    'Sad': {'acousticness': 0.8, 'danceability': 0.2, 'energy': 0.2, 'instrumentalness': 0.3, 'valence': 0.1, 'tempo': 80.0},
    'Surprise': {'acousticness': 0.4, 'danceability': 0.7, 'energy': 0.6, 'instrumentalness': 0.2, 'valence': 0.7, 'tempo': 130.0},
    'Neutral': {'acousticness': 0.5, 'danceability': 0.5, 'energy': 0.5, 'instrumentalness': 0.2, 'valence': 0.5, 'tempo': 110.0},
}

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

# App Header with new styling
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #9b7dd1; font-size: 3em; margin-bottom: 10px; text-shadow: 0px 2px 4px rgba(0,0,0,0.5);">🎵 Aural Prism Song Finder</h1>
    <p style="font-size: 1.2em; color: #d0d0d0;">Discover your perfect soundtrack based on mood, genre, and audio preferences</p>
</div>
""", unsafe_allow_html=True)

# Main container with updated card styling
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3 class='accent-text'>🎛️ Customize Your Preferences</h3>", unsafe_allow_html=True)
        
        # Year range slider with improved styling
        st.markdown("**📅 Select the year range**")
        start_year, end_year = st.slider(
            '', 
            1990, 2019, (2015, 2019),
            label_visibility="collapsed"
        )
        
        # Audio feature sliders
        st.markdown("**🎚️ Adjust audio features**")
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5, help="How acoustic the song is (1.0 = very acoustic)")
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5, help="How suitable the song is for dancing")
        energy = st.slider('Energy', 0.0, 1.0, 0.5, help="The intensity and activity of the song")
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.2, help="The amount of vocals in the song (1.0 = no vocals)")
        valence = st.slider('Valence', 0.0, 1.0, 0.5, help="The musical positiveness (1.0 = happy, cheerful)")
        tempo = st.slider('Tempo (BPM)', 0.0, 244.0, 110.0, help="The speed or pace of the song")
    
    with col2:
        st.markdown("<h3 class='accent-text'>😊 Set Your Mood</h3>", unsafe_allow_html=True)
        
        # Emotion input selection
        emotion_input_type = st.radio(
            "How would you like to set your mood?",
            ["Select emotion", "Describe your mood"],
            index=0
        )
        
        if emotion_input_type == "Select emotion":
            selected_emotion = st.selectbox(
                'Choose an emotion:',
                list(emotion_presets.keys()),
                index=list(emotion_presets.keys()).index('Neutral'),
                help="Select an emotion to automatically adjust audio features"
            )
            
            # Display emotion icon
            emotion_icons = {
                'Angry': '😠',
                'Disgust': '🤢',
                'Fear': '😨',
                'Happy': '😊',
                'Sad': '😢',
                'Surprise': '😲',
                'Neutral': '😐'
            }
            st.markdown(f"<h3 style='text-align: center; font-size: 3em;'>{emotion_icons[selected_emotion]}</h3>", unsafe_allow_html=True)
            
            # Apply emotion presets
            acousticness = emotion_presets[selected_emotion]['acousticness']
            danceability = emotion_presets[selected_emotion]['danceability']
            energy = emotion_presets[selected_emotion]['energy']
            instrumentalness = emotion_presets[selected_emotion]['instrumentalness']
            valence = emotion_presets[selected_emotion]['valence']
            tempo = emotion_presets[selected_emotion]['tempo']
            
        else:
            user_text = st.text_area(
                "How are you feeling today?",
                height=100,
                placeholder="Describe your mood (e.g., 'I feel excited and energetic today!')",
                help="Type how you're feeling and we'll detect the best music for your mood"
            )
            
            analyze_button = st.button("Analyze My Mood", type="primary")
            
            if analyze_button and user_text:
                detected_emotion = advanced_emotion_detection(user_text)
                st.success(f"Detected mood: **{detected_emotion}**")
                selected_emotion = detected_emotion
                
                # Apply emotion presets
                acousticness = emotion_presets[selected_emotion]['acousticness']
                danceability = emotion_presets[selected_emotion]['danceability']
                energy = emotion_presets[selected_emotion]['energy']
                instrumentalness = emotion_presets[selected_emotion]['instrumentalness']
                valence = emotion_presets[selected_emotion]['valence']
                tempo = emotion_presets[selected_emotion]['tempo']
            else:
                selected_emotion = 'Neutral'
        
        # Genre selection with enhanced styling
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**🎶 Select your preferred genre**")
        genre = st.selectbox(
            '',
            genre_names,
            index=genre_names.index("Pop"),
            label_visibility="collapsed"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Generate recommendations
test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
tracks_per_page = 6
tracks = []
for uri in uris:
    track = """<div class="spotify-iframe"><iframe src="https://open.spotify.com/embed/track/{}" width="100%" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe></div>""".format(uri)
    tracks.append(track)

if 'previous_inputs' not in st.session_state:
    st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat

current_inputs = [genre, start_year, end_year] + test_feat
if current_inputs != st.session_state['previous_inputs']:
    if 'start_track_i' in st.session_state:
        st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

if 'start_track_i' not in st.session_state:
    st.session_state['start_track_i'] = 0

# Recommendations section with new styling
st.markdown("""
<div style="margin-top: 30px;">
    <h2 style="color: #9b7dd1; border-bottom: 2px solid #9b7dd1; padding-bottom: 10px; text-shadow: 0px 1px 3px rgba(0,0,0,0.3);">🎧 Your Personalized Recommendations</h2>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if st.button("🔀 Show More Recommendations", type="primary"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page
    
    current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    
    if st.session_state['start_track_i'] < len(tracks):
        # Create two columns for track display
        cols = st.columns(2)
        
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            with cols[i % 2]:
                components.html(track, height=400)
                
                # Audio features radar chart with updated styling
                with st.expander("📊 Track Audio Features"):
                    df = pd.DataFrame(dict(
                        r=audio[:5],
                        theta=audio_feats[:5]
                    ))
                    fig = px.line_polar(
                        df, 
                        r='r', 
                        theta='theta', 
                        line_close=True,
                        color_discrete_sequence=['#9b7dd1'],
                        template="plotly_dark"
                    )
                    fig.update_layout(
                        height=350,
                        width=350,
                        paper_bgcolor='rgba(30,30,30,0.0)',
                        plot_bgcolor='rgba(30,30,30,0.0)',
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                linecolor='rgba(150,150,150,0.3)',
                                gridcolor='rgba(150,150,150,0.2)'
                            ),
                            angularaxis=dict(
                                linecolor='rgba(150,150,150,0.3)',
                                gridcolor='rgba(150,150,150,0.2)'
                            ),
                            bgcolor='rgba(30,30,30,0.5)'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("🎉 You've reached the end of recommendations! Try adjusting your filters for more songs.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with updated styling
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: #808080; font-size: 0.9em;">
    <p>Powered by Spotify data and machine learning</p>
    <p>Made Aural Prism ❤️ for music lovers</p>
</div>
""", unsafe_allow_html=True)