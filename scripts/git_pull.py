"""
Script para baixar atualizações do GitHub
Uso: python git_pull.py
"""
import subprocess
import sys


def run_cmd(cmd, check=True):
    """Executa comando e retorna output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"Erro no comando: {cmd}")
        return False
    return True


def main():
    print("=" * 50)
    print("  GIT PULL - Baixando do GitHub")
    print("=" * 50)
    print()

    # 1. Ver status local
    print("[1/3] Verificando status local...")
    run_cmd("git status", check=False)

    # 2. Fetch para ver o que tem de novo
    print("\n[2/3] Buscando atualizações...")
    run_cmd("git fetch origin", check=False)

    # 3. Pull
    print("\n[3/3] Baixando alterações...")
    if not run_cmd("git pull origin main", check=False):
        # Tentar com master se main falhar
        print("Tentando branch master...")
        run_cmd("git pull origin master", check=False)

    print()
    print("=" * 50)
    print("  Concluido!")
    print("=" * 50)


if __name__ == "__main__":
    main()
