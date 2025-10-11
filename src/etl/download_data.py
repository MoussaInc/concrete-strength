# src/etl/download_data.py

import os
import sys
import io
import re
import zipfile
import argparse
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

# --- NOUVELLES DÉPENDANCES ---
try:
    import yaml
except ImportError:
    print("FATAL: Le module 'pyyaml' est requis. Installez-le avec 'pip install pyyaml'.")
    sys.exit(1)

# --- Configuration et chemins (basés sur la racine du projet) ---
# Chemin vers le dossier racine du projet (deux niveaux au-dessus de src/etl)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"

# Charger la configuration
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"FATAL: Fichier de configuration manquant: {CONFIG_PATH}. Assurez-vous d'avoir créé 'config/datasets.yaml'.")
    sys.exit(1)

DATASETS = CONFIG.get("datasets", {})
RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["raw_dir"]
DOCS_DIR = PROJECT_ROOT / CONFIG["paths"]["docs_dir"]
LOG_DIR = PROJECT_ROOT / CONFIG["paths"]["log_dir"]

# --- Setup Logging ---
def setup_logging(level=logging.INFO):
    """
    Configuration du système de logging
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Pour éviter les handlers multiples si le script est rechargé
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_handler = logging.FileHandler(LOG_DIR / "data_download.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream Handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

logger = setup_logging()

# --- Classes d'Extraction Spécialisées  ---

class MendeleyPDFExtractor:
    """
    Gestion de l'extraction très spécifique du PDF du dataset Mendeley (IIT Bhubaneswar)
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.expected_columns = 17
    
    def clean_row(self, row: List) -> List[str]:
        """
        Nettoie d'une ligne de données
        """

        return [str(cell or '').strip().replace('\n', ' ') for cell in row if cell not in (None, '')]

    def is_header_row(self, row: List[str]) -> bool:
        """
        Détecte si une ligne est un en-tête pour les pages suivantes
        """

        header_keywords = ['serial', 'cement', 'flyash', 'ggbs', 'water', 'age', 'cs', 'tcm']
        row_str = ' '.join(row).lower()
        return any(keyword in row_str for keyword in header_keywords)

    def __call__(self, pdf_path: Path, out_csv_path: Path) -> bool:
        """
        Exécute l'extraction avancée
        """

        try:
            import pdfplumber
            
            all_rows = []
            header = None
            
            self.logger.info("🔍 Extraction PDF avancée (format béton IIT Bhubaneswar)")
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    self.logger.debug(f"Traitement page {page_num + 1}")
                    
                    tables = page.extract_tables()
                    if not tables: continue
                    
                    for table in tables:
                        if not table or len(table) < 2: continue
                        
                        # Identification et capture de l'en-tête (première page)
                        if page_num == 0 and not header:
                            # Logique de recherche d'en-tête
                            header_row_index = -1
                            for i, row in enumerate(table):
                                row_str = ' '.join(str(cell or '') for cell in row).lower()
                                if self.is_header_row([row_str]):
                                    header = [str(cell or '').strip() for cell in row]
                                    header_row_index = i
                                    self.logger.info(f"En-tête identifié (cols: {len(header)}) : {header[:5]}...")
                                    break
                            
                            # Ajouter les données après l'en-tête
                            if header_row_index != -1:
                                for data_row in table[header_row_index+1:]:
                                    cleaned = self.clean_row(data_row)
                                    if len(cleaned) >= self.expected_columns - 2:
                                        all_rows.append(cleaned)
                        
                        # Pages suivantes : seulement les données
                        elif header:
                            for row in table:
                                cleaned = self.clean_row(row)
                                if (len(cleaned) >= self.expected_columns - 2 and 
                                    not self.is_header_row(cleaned)):
                                    all_rows.append(cleaned)
            
            # Validation et création du DataFrame
            if header and all_rows:
                return self.create_final_dataframe(header, all_rows, out_csv_path)
            else:
                self.logger.warning("En-tête ou données manquantes après l'extraction avancée")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur extraction PDF avancée (Mendeley): {e}")
            return False

    def create_final_dataframe(self, header: List[str], all_rows: List[List[str]], out_csv_path: Path) -> bool:
        """
        Crée le DataFrame final et le sauvegarde (similaire à l'original)
        """

        try:
            # Nettoyage des données (implémenter la logique de normalisation et de conversion types)
            max_cols = len(header)
            normalized_data = []
            
            for row in all_rows:
                row = [c for c in row if c] # Enlève les None ou vides
                if len(row) < max_cols:
                    row = row + [''] * (max_cols - len(row))
                elif len(row) > max_cols:
                    row = row[:max_cols]
                
                if any(cell.strip() for cell in row):
                    normalized_data.append(row)
            
            if not normalized_data:
                self.logger.warning("Aucune donnée valide après normalisation")
                return False

            df = pd.DataFrame(normalized_data, columns=header)
            df.columns = [re.sub(r'\s+', ' ', col.strip()) for col in df.columns]

            # Tentative de conversion numérique sur toutes les colonnes sauf 'Serial No' (si elle existe)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass # Laisser comme string si échec

            df = df.dropna(thresh=int(max_cols * 0.5)) # Lignes avec au moins 50% des valeurs
            
            if len(df) > 0:
                df.to_csv(out_csv_path, index=False)
                self.logger.info(f"✅ PDF -> CSV réussi : {len(df)} lignes, {len(df.columns)} colonnes")
                return True
            else:
                self.logger.warning("DataFrame vide après nettoyage")
                return False
        except Exception as e:
            self.logger.error(f"❌ Erreur création DataFrame : {e}")
            return False

# --- Classe Principale du Téléchargeur ---

class DataDownloader:
    """
    Classe principale pour le téléchargement, l'extraction et la conversion des datasets.
    """
    
    def __init__(self, datasets_config: Dict, raw_dir: Path, docs_dir: Path, logger: logging.Logger):
        self.datasets = datasets_config
        self.raw_dir = raw_dir
        self.docs_dir = docs_dir
        self.logger = logger
        self.mendeley_extractor = MendeleyPDFExtractor(logger)
        
        # Vérification des dépendances
        self.deps = self._check_optional_dependencies()

    # --- Utilitaires ---
    
    def _check_optional_dependencies(self) -> Dict[str, bool]:
        """
        Vérification et gestion l'installation des dépendances optionnelles (intégré à la classe)
        """

        optional_deps = {
            'tabula': False, 
            'pdfplumber': False, 
            'openpyxl': False, 
            'xlrd': False
        }
        
        missing = []
        for import_name in optional_deps.keys():
            try:
                __import__(import_name)
                optional_deps[import_name] = True
            except ImportError:
                missing.append(import_name)
                
        if missing:
            self.logger.warning(f"Dépendances manquantes : {', '.join(missing)}")
            
            # --- Ajouter l'option d'installation (CLI) ---
            response = input("Souhaitez-vous installer les dépendances manquantes (y/N)? ").strip().lower()
            if response == 'y':
                try:
                    import subprocess
                    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)
                    self.logger.info("Dépendances installées.")
                    # Re-vérification après installation
                    for import_name in missing:
                        try:
                            __import__(import_name)
                            optional_deps[import_name] = True
                        except ImportError:
                            pass
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Erreur lors de l'installation : {e}")
                    
        return optional_deps
        
    def _save_bytes(self, path: Path, content: bytes):
        """
        Sauvegarde des bytes dans un fichier
        """

        with open(path, "wb") as f:
            f.write(content)
        self.logger.debug(f"Fichier sauvegardé : {path.name}")

    def _should_download(self, name: str, meta: dict, force: bool = False) -> bool:
        """
        Vérifie si le téléchargement est nécessaire
        """

        if force:
            return True
            
        base = meta["base"]
        expected_files = [
            self.raw_dir / f"{base}.csv",
            self.raw_dir / f"{base}_from_pdf.csv",
            self.docs_dir / f"{base}.pdf"
        ]
        
        existing = [f for f in expected_files if f.exists()]
        if existing:
            self.logger.warning(f"Fichiers existants pour {name}. Retélécharger ? (o/N)")
            response = input("Retélécharger ? (o/N): ").strip().lower()
            return response in ('o', 'oui', 'y', 'yes')
        return True

    # --- Téléchargement et Détection ---

    def _get_filename_from_headers(self, resp, fallback: str) -> str:
        """
        Extrait le filename depuis les headers HTTP
        """

        cd = resp.headers.get("Content-Disposition", "")
        match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if match:
            return match.group(1)
        tail = resp.url.split("?")[0].rstrip("/").split("/")[-1]
        return tail or fallback

    def _detect_ext(self, name_or_url: str, content_type: str) -> str:
        """
        Détecte l'extension du fichier (Identique à l'original, très bien)
        """

        name_lower = (name_or_url or "").lower()
        if name_lower.endswith((".csv", ".xlsx", ".xls", ".zip", ".pdf")):
            return name_lower.split('.')[-1]
        
        ct = (content_type or "").lower()
        if "zip" in ct: return "zip"
        if "pdf" in ct: return "pdf"
        if "excel" in ct or "spreadsheetml" in ct: return "xlsx"
        if "csv" in ct or "text/plain" in ct: return "csv"
        
        return ""

    def download(self, url: str, fallback_name: str) -> tuple:
        """
        Télécharge un fichier avec gestion d'erreurs (Identique à l'original)
        """

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        
        filename = self._get_filename_from_headers(resp, fallback=fallback_name)
        ext = self._detect_ext(filename or url, resp.headers.get("Content-Type", ""))
        
        self.logger.info(f"Téléchargement réussi : {filename} ({len(resp.content)} bytes, type: {ext})")
        return filename, resp.content, ext, resp.headers.get("Content-Type", "")

    # --- Extraction et Conversion ---

    def _excel_to_csv_bytes(self, excel_bytes: bytes, ext: str) -> bytes:
        """
        Convertion de Excel --> en CSV
        """

        try:
            bio = io.BytesIO(excel_bytes)
            # Utilisation des dépendances pour garantir la lecture
            engine = 'openpyxl' if ext == 'xlsx' and self.deps.get('openpyxl') else ('xlrd' if self.deps.get('xlrd') else None)
            
            if not engine:
                self.logger.error("Moteur de lecture Excel manquant (openpyxl/xlrd).")
                raise ImportError("Moteur Excel manquant")
                
            df = pd.read_excel(bio, engine=engine)
            out = io.StringIO()
            df.to_csv(out, index=False)
            self.logger.info(f"Excel converti en CSV ({len(df)} lignes)")
            return out.getvalue().encode("utf-8")
        except Exception as e:
            self.logger.error(f"Erreur conversion Excel : {e}")
            raise

    def _extract_with_tabula_fallback(self, pdf_path: Path, out_csv_path: Path) -> bool:
        """
        Fallback avec Tabula pour l'extraction PDF (Simplifié)
        """

        if not self.deps.get('tabula'):
            self.logger.warning("Tabula non disponible.")
            return False

        try:
            import tabula
            self.logger.info("Essai Tabula (Stream/Lattice All Pages)")
            
            # Essayer les deux modes les plus communs
            dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, guess=False, stream=True)
            if not dfs:
                 dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, guess=False, lattice=True)
            
            if dfs and len(dfs) > 0:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.dropna(how='all')
                
                if len(combined_df) > 3:
                    combined_df.to_csv(out_csv_path, index=False)
                    self.logger.info(f"Tabula réussi : {len(combined_df)} lignes")
                    return True
                
            return False
        except Exception as e:
            self.logger.debug(f"Tabula échoué : {e}")
            return False
            
    def extract_tables_from_pdf(self, pdf_path: Path, out_csv_path: Path, name: str) -> bool:
        """
        Fonction principale d'extraction PDF avec stratégies
        """

        self.logger.info(f"Extraction PDF : {pdf_path.name}")
        
        # Extraction Avancée (Spécifique Mendeley)
        if name.lower() == "mendeley" and self.deps.get('pdfplumber'):
            if self.mendeley_extractor(pdf_path, out_csv_path):
                return True

        # Fallback Tabula
        if self._extract_with_tabula_fallback(pdf_path, out_csv_path):
            return True
        
        # Dernier recours : pdfplumber basique (non implémenté ici pour concision, car la logique avancée est dans MendeleyExtractor)
        
        self.logger.error("Aucune méthode d'extraction PDF n'a fonctionné")
        return False

    # --- Gestion des Archives ZIP ---

    def _pick_inner_name(self, names, regex=None, want_table=True):
        # (Fonction de sélection des fichiers dans le ZIP, inchangée car efficace)
        if regex:
            pattern = re.compile(regex)
            names = [n for n in names if pattern.search(n)]
        
        priority = [r"(?i)\.csv$", r"(?i)\.xlsx$", r"(?i)\.xls$"] if want_table else [r"(?i)\.pdf$"]
        candidates = [n for n in names if re.search(r"(?i)\.(csv|xlsx?|pdf)$", n)]
        
        def score(n):
            for i, p in enumerate(priority):
                if re.search(p, n): return i
            return 99
        
        candidates = sorted(candidates, key=score)
        return candidates[0] if candidates else None

    def _extract_from_zip(self, zip_bytes: bytes, base: str, inner_regex: Optional[str], prefer_pdf=False, name="") -> Dict[str, Any]:
        """
        Extrait le contenu d'une archive ZIP
        """

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            self.logger.info(f"Contenu ZIP : {names[:5]}...")

            if prefer_pdf:
                # Priorité PDF (Mendeley)
                pdf_target = self._pick_inner_name(names, regex=None, want_table=False)
                if pdf_target:
                    return self._extract_pdf_from_zip(zf, pdf_target, base, name)
            
            # Priorité Table (CSV/Excel)
            table_target = self._pick_inner_name(names, regex=inner_regex, want_table=True)
            if table_target:
                return self._extract_table_from_zip(zf, table_target, base)
            
            # Fallback PDF (si on n'a pas déjà fait le PDF)
            if not prefer_pdf:
                pdf_target = self._pick_inner_name(names, regex=None, want_table=False)
                if pdf_target:
                    return self._extract_pdf_from_zip(zf, pdf_target, base, name)

            # Aucun fichier exploitable
            return self._save_zip_for_inspection(zip_bytes, base, names)

    def _extract_table_from_zip(self, zf, table_target: str, base: str) -> Dict[str, Any]:
        """
        Extrait et convertit un tableau depuis ZIP
        """

        with zf.open(table_target, "r") as f:
            data = f.read()
        
        out_csv = self.raw_dir / f"{base}.csv"
        
        if table_target.lower().endswith(".csv"):
            self._save_bytes(out_csv, data)
            return {"kind": "csv", "path": out_csv}
        else:
            ext = "xlsx" if table_target.lower().endswith(".xlsx") else "xls"
            csv_bytes = self._excel_to_csv_bytes(data, ext)
            self._save_bytes(out_csv, csv_bytes)
            return {"kind": "excel", "path": out_csv}

    def _extract_pdf_from_zip(self, zf, pdf_target: str, base: str, name: str) -> Dict[str, Any]:
        """
        Extrait un PDF depuis ZIP et tente l'extraction des tableaux
        """

        with zf.open(pdf_target, "r") as f:
            pdf_data = f.read()
        
        out_pdf = self.docs_dir / f"{base}.pdf"
        self._save_bytes(out_pdf, pdf_data)
        
        # Tenter l'extraction des tableaux
        out_csv = self.raw_dir / f"{base}_from_pdf.csv"
        if self.extract_tables_from_pdf(out_pdf, out_csv, name):
            return {"kind": "pdf->csv", "path": out_csv}
        else:
            self.logger.info(f"PDF sauvegardé (extraction auto échouée) : {out_pdf.name}")
            return {"kind": "pdf", "path": out_pdf}

    def _save_zip_for_inspection(self, zip_bytes: bytes, base: str, names: List[str]) -> Dict[str, Any]:
        """
        Sauvegarde l'archive pour inspection manuelle
        """

        zip_out = self.raw_dir / f"{base}_original.zip"
        self._save_bytes(zip_out, zip_bytes)
        
        manifest = self.raw_dir / f"{base}_zip_manifest.txt"
        with open(manifest, "w", encoding="utf-8") as mf:
            mf.write("\n".join(names))
        
        self.logger.warning(f"Aucun fichier exploitable, archive sauvegardée : {zip_out.name}")
        return {"kind": "none", "path": zip_out}

    # --- Pipeline principal ---

    def process_dataset(self, name: str, meta: dict, force: bool = False):
        """
        Traite un dataset complet
        """

        try:
            if not self._should_download(name, meta, force):
                self.logger.info(f"Dataset {name} ignoré (fichiers existants)")
                return True
                
            base = meta["base"]
            url = meta["url"]
            inner_regex = meta.get("inner_regex")
            
            self.logger.info(f"Traitement dataset {name.upper()}...")
            filename, content, ext, ctype = self.download(url, fallback_name=f"{base}")

            if ext == "csv":
                out_csv = self.raw_dir / f"{base}.csv"
                self._save_bytes(out_csv, content)
                self.logger.info(f"CSV sauvegardé : {out_csv.name}")
                return True

            elif ext in ("xls", "xlsx"):
                out_csv = self.raw_dir / f"{base}.csv"
                csv_bytes = self._excel_to_csv_bytes(content, ext)
                self._save_bytes(out_csv, csv_bytes)
                self.logger.info(f"Excel converti en CSV : {out_csv.name}")
                return True

            elif ext == "pdf":
                out_pdf = self.docs_dir / f"{base}.pdf"
                self._save_bytes(out_pdf, content)
                out_csv = self.raw_dir / f"{base}_from_pdf.csv"
                if self.extract_tables_from_pdf(out_pdf, out_csv, name):
                    self.logger.info(f"PDF -> CSV réussi : {out_csv.name}")
                else:
                    self.logger.info(f"PDF sauvegardé (extraction auto échouée) : {out_pdf.name}")
                return True

            elif ext == "zip":
                prefer_pdf = (name.lower() == "mendeley")
                result = self._extract_from_zip(content, base=base, inner_regex=inner_regex, prefer_pdf=prefer_pdf, name=name)
                
                kind = result["kind"]
                path = result["path"].name if isinstance(result["path"], Path) else result["path"]
                
                if kind in ("csv", "excel", "pdf->csv"):
                    self.logger.info(f"Extraction ZIP réussie : {path}")
                elif kind == "pdf":
                    self.logger.info(f"ZIP -> PDF : {path}")
                else:
                    self.logger.warning(f"ZIP nécessite inspection : {path}")
                return True

            else:
                self.logger.error(f"Format non géré : {ext} (Content-Type: {ctype})")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur traitement dataset {name} : {e}")
            return False

# --- Interface utilisateur et main ---

def interactive_mode(datasets: Dict) -> List[str]:
    """
    Mode interactif pour la sélection des datasets
    """

    print("\n=== Téléchargement des datasets Béton ===")
    
    keys = list(datasets.keys())
    for i, key in enumerate(keys):
        print(f"[{i+1}] {key.capitalize()} ({datasets[key].get('notes', 'Pas de note')})")
    print(f"[{len(keys) + 1}] Tous les datasets")
    print(f"[{len(keys) + 2}] Quitter")

    choice = input(f"Entrez 1 à {len(keys) + 2} : ").strip()
    
    if choice.isdigit():
        choice_int = int(choice)
        if 1 <= choice_int <= len(keys):
            return [keys[choice_int - 1]]
        elif choice_int == len(keys) + 1:
            return keys
        elif choice_int == len(keys) + 2:
            return []
            
    print("Choix invalide.")
    return interactive_mode(datasets)


def main():
    """
    Fonction principale
    """

    parser = argparse.ArgumentParser(description="Télécharge les datasets Béton à partir de config/datasets.yaml")
    parser.add_argument('--dataset', nargs='+', choices=list(DATASETS.keys()) + ['all'], help='Datasets à télécharger (uci, figshare, mendeley, all)')
    parser.add_argument('--force', action='store_true', help='Forcer le retéléchargement')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Niveau de logging')
    parser.add_argument('--interactive', action='store_true', help='Mode interactif')
    
    args = parser.parse_args()
    
    # Reconfiguration du logger si l'argument log-level est utilisé
    global logger
    logger = setup_logging(getattr(logging, args.log_level))
    
    # Création des répertoires
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Répertoires créés/vérifiés")

    # Initialisation du téléchargeur
    downloader = DataDownloader(DATASETS, RAW_DIR, DOCS_DIR, logger)
    
    # Déterminer les datasets à traiter
    datasets_to_process = []
    
    if args.interactive:
        datasets_to_process = interactive_mode(DATASETS)
    elif args.dataset:
        if 'all' in args.dataset:
            datasets_to_process = list(DATASETS.keys())
        else:
            datasets_to_process = args.dataset
    else:
        logger.info("Utilisation : python download_data.py --dataset uci figshare --force")
        logger.info("Ou : python download_data.py --interactive")
        return
    
    if not datasets_to_process:
        logger.info("Aucun dataset sélectionné, arrêt.")
        return
    
    # Traitement
    logger.info(f"Début du traitement des datasets : {datasets_to_process}")
    success_count = 0
    
    for dataset in datasets_to_process:
        if downloader.process_dataset(dataset, DATASETS[dataset], args.force):
            success_count += 1
    
    # Résumé
    logger.info(f"\nRÉSUMÉ : {success_count}/{len(datasets_to_process)} datasets traités avec succès")

if __name__ == "__main__":
    main()