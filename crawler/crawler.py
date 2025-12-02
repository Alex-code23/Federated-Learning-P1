import requests
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from collections import deque
import os, json

def allowed_by_robots(url, user_agent="MyCrawler/1.0"):
    """Vérifie si le site autorise le crawl de cette URL."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return False

def crawl_website(start_url, max_pages=200):
    """Crawl toutes les pages internes d’un site (respecte robots.txt)."""
    domain = urlparse(start_url).netloc
    visited = set()
    queue = deque([start_url])
    all_pages = []

    user_agent = "MyCrawler/1.0"
    headers = {"User-Agent": user_agent}

    # Test robots.txt pour URL de départ
    if not allowed_by_robots(start_url, user_agent):
        print(f"❌ robots.txt interdit l'accès à : {start_url}")
        return []

    print(f"📡 Crawling du site : {domain}")

    while queue and len(visited) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue

        if not allowed_by_robots(url, user_agent):
            print(f"⛔ Interdit par robots.txt : {url}")
            continue

        try:
            response = requests.get(url, headers=headers, timeout=5)
        except:
            print(f"⚠️ Erreur requête : {url}")
            continue

        if response.status_code != 200:
            continue

        visited.add(url)
        all_pages.append(url)

        print(f"✅ Page trouvée : {url}")

        # Analyse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Extraction des liens internes
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)

            # Filtre : interne au domaine
            if urlparse(full_url).netloc == domain:
                if full_url not in visited:
                    queue.append(full_url)

    print("\n🔍 Crawl terminé")
    print(f"📄 Nombre total de pages trouvées : {len(all_pages)}")

    return all_pages

def save_crawled_pages(start_url, pages):
    """Sauvegarde la liste des pages crawlées dans un fichier JSON."""
    domain = urlparse(start_url).netloc
    filename = domain.replace('.', '_') + '.json'
    
    output_dir = "crawled_sites"
    # Le dossier de sortie est relatif à l'emplacement du script
    output_dir = os.path.join(os.path.dirname(__file__), "crawled_sites")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
        
    print(f"💾 Pages sauvegardées dans : {output_path}")

# ------------------------
# Exemple d'utilisation
# ------------------------
if __name__ == "__main__":
    # Liste des sites à crawler
    sites_to_crawl = [
        "https://books.toscrape.com/",
        "https://quotes.toscrape.com/",
        "https://www.python.org/",
        "https://www.djangoproject.com/",
        "https://flask.palletsprojects.com/",
        "https://numpy.org/",
        "https://pandas.pydata.org/"
    ]

    MAX_PAGES_PER_SITE = 70

    for site_url in sites_to_crawl:
        pages = crawl_website(site_url, max_pages=MAX_PAGES_PER_SITE)
        if pages:
            save_crawled_pages(site_url, pages)
        print("-" * 40)
