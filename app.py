import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set Streamlit page configuration
st.set_page_config(
    page_title="Cluster BBC News Articles",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Set background color and title styles
st.markdown(
    """
    <style>
    body {
        background-color: #f2f2f2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #808080;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stMarkdown h1, .stMarkdown h2 {
        color: #23558A;
    }
    .stButton {
        background-color: #23558A;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton:hover {
        background-color: #808080;
    }
    .stSlider {
        width: 80%;
    }
    .stSlider .stSliderValue {
        color: #808080;
    }
    .stMarkdown p {
        font-size: 16px;
    }
    .cluster-container {
        background-color: #23558A;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .cluster-header {
        color: #808080;
        font-size: 24px;
        margin-top: 10px;
    }
    .cluster-terms {
        font-weight: bold;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .article-title {
        font-weight: bold;
        color: #23558A;
    }
    .article-link {
        color: #ffffff;
        text-decoration: none;
        transition: color 0.3s;
    }
    .article-link:hover {
        color: #eeeeee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def scrape_news():
    url = 'https://www.bbc.com/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_tags = soup.find_all('a', {'class': 'gs-c-promo-heading'})

    articles = []
    for tag in article_tags[:20]:  # Limit to 20 articles for simplicity
        title = tag.text.strip()
        link = tag['href']
        if not link.startswith('http'):
            link = f'https://www.bbc.com/news{link}'
        articles.append({'title': title, 'link': link})
   
    return articles

def cluster_articles(articles, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([article['title'] for article in articles])

    if X.shape[0] < n_clusters:
        n_clusters = X.shape[0]
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
   
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
   
    clusters = {}
    for i in range(n_clusters):
        cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]
        clusters[i] = {'articles': [], 'terms': ', '.join(cluster_terms)}
       
    for i, label in enumerate(kmeans.labels_):
        clusters[label]['articles'].append(articles[i])
       
    return clusters

# Streamlit UI
st.title('Cluster BBC News Articles')

# Number input for selecting the number of clusters
n_clusters = st.number_input('Select number of clusters', min_value=2, max_value=25, value=5, step=1)

# Button to trigger scraping and clustering
if st.button('Scrape and Cluster Articles', key="button"):
    articles = scrape_news()
    if articles:
        # Perform clustering
        clusters = cluster_articles(articles, n_clusters=n_clusters)

        # Display clustered articles
        for cluster_id, cluster_info in clusters.items():
            st.markdown(f"<div class='cluster-container'>", unsafe_allow_html=True)
            st.markdown(f"<div class='cluster-header'>Cluster {cluster_id + 1}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='cluster-terms'>Top terms: {cluster_info['terms']}</p>", unsafe_allow_html=True)

            # Create two columns for titles and links
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<p class='article-title'>Titles</p>", unsafe_allow_html=True)
                for article in cluster_info['articles']:
                    st.write(article['title'])

            with col2:
                st.markdown("<p class='article-title'>Links</p>", unsafe_allow_html=True)
                for article in cluster_info['articles']:
                    st.markdown(f"<a class='article-link' href='{article['link']}' target='_blank'>{article['link']}</a>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No articles found. Please check the URL or try a different site.")



