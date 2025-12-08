import requests
import time
import json
import os
from bs4 import BeautifulSoup
import numpy as np
from bs4 import XMLParsedAsHTMLWarning
import warnings

# -----------------------
# Collecte des stats web
# -----------------------
def collect_web_stats(url):
    """Analyse une URL et retourne un dictionnaire de statistiques et une étiquette."""
    user_agent = "MyResearchBot/1.0"
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None


        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        soup = BeautifulSoup(response.text, "html.parser")

        text_length = len(soup.get_text())
        num_links = len(soup.find_all("a"))
        num_images = len(soup.find_all("img"))
        num_paragraphs = len(soup.find_all("p"))
        num_scripts = len(soup.find_all("script"))
        num_headers = sum(len(soup.find_all(f"h{i}")) for i in range(1, 7))
        content_size_kb = round(len(response.content) / 1024, 2)

        # --- Logique de classification par ratios ---
        link_density = (num_links / text_length * 500) if text_length > 0 else float('inf')
        text_to_content_ratio = text_length / (content_size_kb * 1024) if content_size_kb > 0 else 0

        if link_density > 10 and num_links > 2:
            label = "navigation-heavy"
        elif num_scripts > 7:
            label = "script-heavy"
        elif text_to_content_ratio > 0.3 and text_length > 300:
            label = "text-heavy"
        else:
            label = "other"

        return {
            "url": url,
            "text_length": text_length,
            "num_links": num_links,
            "num_images": num_images,
            "num_paragraphs": num_paragraphs,
            "num_scripts": num_scripts,
            "num_headers": num_headers,
            "content_size_kb": content_size_kb,
            "label": label
        }

    except Exception as e:
        print(f"Erreur sur {url} : {e}")
        return None

# -----------------------
# Préparation des données X, y
# -----------------------
def prepare_Xy(data):
    """Convertit les données brutes en vecteurs de caractéristiques (X) et étiquettes (y)."""
    X = []
    y = []
    for d in data:
        features = [
            d["text_length"],
            d["num_links"],
            d["num_images"],
            d["num_paragraphs"],
            d["num_scripts"],
            d["num_headers"],
            d["content_size_kb"]
        ]
        X.append(features)
        y.append(d["label"])
    return X, y

# -----------------------
# Script principal
# -----------------------
if __name__ == "__main__":
    CRAWLED_SITES_DIR = "crawler/crawled_sites"
    PROCESSED_DATA_DIR = "crawler/processed_data"
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if not os.path.exists(CRAWLED_SITES_DIR):
        print(f"Le dossier '{CRAWLED_SITES_DIR}' n'existe pas. Exécutez d'abord crawler.py.")
    else:
        for filename in os.listdir(CRAWLED_SITES_DIR):
            if filename.endswith(".json"):
                site_name = filename.replace(".json", "")
                print(f"--- Traitement de {site_name} ---")
                
                input_path = os.path.join(CRAWLED_SITES_DIR, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    urls = json.load(f)
                
                site_stats = []
                for url in urls:
                    stats = collect_web_stats(url)
                    if stats:
                        site_stats.append(stats)
                
                X, y = prepare_Xy(site_stats)
                
                output_path = os.path.join(PROCESSED_DATA_DIR, f"{site_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({"X": X, "y": y}, f, indent=4)
                    
                print(f"✅ Données traitées et sauvegardées dans {output_path}\n")
    
    print("Traitement terminé.")