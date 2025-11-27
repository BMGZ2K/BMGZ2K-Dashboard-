"""
Script para atualizar o projeto no GitHub
Uso: python git_push.py "mensagem do commit"
"""
import subprocess
import sys
from datetime import datetime


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
    # Mensagem do commit
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        # Mensagem padrão com timestamp
        message = f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    print("=" * 50)
    print("  GIT PUSH - Atualizando GitHub")
    print("=" * 50)
    print(f"Mensagem: {message}")
    print()

    # 1. Ver status
    print("[1/4] Verificando status...")
    run_cmd("git status", check=False)

    # 2. Adicionar todos os arquivos
    print("\n[2/4] Adicionando arquivos...")
    if not run_cmd("git add -A"):
        return

    # 3. Commit
    print("\n[3/4] Criando commit...")
    # Escapar aspas na mensagem
    message_escaped = message.replace('"', '\\"')
    if not run_cmd(f'git commit -m "{message_escaped}"', check=False):
        print("Nenhuma alteracao para commit ou erro no commit")

    # 4. Push
    print("\n[4/4] Enviando para GitHub...")
    if not run_cmd("git push origin main"):
        # Tentar com master se main falhar
        print("Tentando branch master...")
        run_cmd("git push origin master", check=False)

    print()
    print("=" * 50)
    print("  Concluido!")
    print("=" * 50)


if __name__ == "__main__":
    main()
