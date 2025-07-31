import sys
import os
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

def run_script(script_path):
    if not os.path.exists(script_path):
        print(f"[SKIP] Script introuvable : {script_path}")
        return
    print(f"\n=== Exécution de {script_path} ===\n")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Le script {script_path} a retourné un code erreur {result.returncode}")

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    run_script(os.path.join(base, "connecteurs", "connecteurs.py"))
    run_script(os.path.join(base, "Analyse_Exploratoire_des_donnes", "EDA.py"))
    run_script(os.path.join(base, "Preprocessing", "preprocessing.py"))
    run_script(os.path.join(base, "Modéles_Machine_Learning", "Modeling.py"))

    print("\n=== Lancement de l'application Flask ===\n")
    os.chdir(os.path.join(base, "api"))
    os.system(f"{sys.executable} app.py")

if __name__ == "__main__":
    main()