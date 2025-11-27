"""
Utility functions for the trading system.
Includes atomic file operations to prevent race conditions.

VERSÃO: 2.0
- Backup automático de state
- Rotação de backups (manter últimos N)
- Rotação de logs
"""
import json
import os
import tempfile
import shutil
import logging
import time
import glob
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# Configurações de backup
BACKUP_DIR = 'state/backups'
MAX_BACKUPS = 10  # Manter últimos 10 backups

def save_json_atomic(filepath: str, data: Any, indent: int = 2):
    """
    Save JSON data atomically to avoid partial writes or race conditions.
    Writes to a temp file first, then renames it to the target file.
    """
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    # Create temp file in the same directory to ensure atomic rename works
    # (rename across filesystems might not be atomic)
    fd, temp_path = tempfile.mkstemp(dir=dir_name, text=True)
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
            
        # Atomic rename
        shutil.move(temp_path, filepath)
        
    except Exception as e:
        logger.error(f"Error saving JSON atomically to {filepath}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e

def load_json_safe(filepath: str, default: Any = None, retries: int = 3, delay: float = 0.1) -> Any:
    """
    Load JSON data safely with retries.
    """
    if default is None:
        default = {}
        
    if not os.path.exists(filepath):
        return default
        
    for i in range(retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            if i == retries - 1:
                logger.error(f"JSON decode error in {filepath}")
                return default
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default

    return default


def backup_state_file(filepath: str) -> bool:
    """
    Cria backup de um arquivo de state.
    Mantém apenas os últimos MAX_BACKUPS backups.

    Args:
        filepath: Caminho do arquivo para fazer backup

    Returns:
        True se backup foi criado com sucesso
    """
    if not os.path.exists(filepath):
        return False

    try:
        # Criar diretório de backup se não existir
        os.makedirs(BACKUP_DIR, exist_ok=True)

        # Gerar nome do backup com timestamp
        basename = os.path.basename(filepath)
        name, ext = os.path.splitext(basename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(BACKUP_DIR, backup_name)

        # Copiar arquivo
        shutil.copy2(filepath, backup_path)
        logger.debug(f"Backup criado: {backup_path}")

        # Limpar backups antigos
        cleanup_old_backups(name, ext)

        return True

    except Exception as e:
        logger.error(f"Erro ao criar backup de {filepath}: {e}")
        return False


def cleanup_old_backups(basename: str, ext: str):
    """
    Remove backups antigos, mantendo apenas os últimos MAX_BACKUPS.

    Args:
        basename: Nome base do arquivo (sem extensão)
        ext: Extensão do arquivo
    """
    try:
        pattern = os.path.join(BACKUP_DIR, f"{basename}_*{ext}")
        backups = sorted(glob.glob(pattern), reverse=True)

        # Remover backups extras
        for old_backup in backups[MAX_BACKUPS:]:
            try:
                os.remove(old_backup)
                logger.debug(f"Backup antigo removido: {old_backup}")
            except Exception as e:
                logger.warning(f"Erro ao remover backup antigo {old_backup}: {e}")

    except Exception as e:
        logger.warning(f"Erro ao limpar backups antigos: {e}")


def restore_latest_backup(filepath: str) -> bool:
    """
    Restaura o backup mais recente de um arquivo.

    Args:
        filepath: Caminho do arquivo para restaurar

    Returns:
        True se restauração foi bem sucedida
    """
    try:
        basename = os.path.basename(filepath)
        name, ext = os.path.splitext(basename)
        pattern = os.path.join(BACKUP_DIR, f"{name}_*{ext}")
        backups = sorted(glob.glob(pattern), reverse=True)

        if not backups:
            logger.warning(f"Nenhum backup encontrado para {filepath}")
            return False

        latest_backup = backups[0]
        shutil.copy2(latest_backup, filepath)
        logger.info(f"Estado restaurado de: {latest_backup}")
        return True

    except Exception as e:
        logger.error(f"Erro ao restaurar backup de {filepath}: {e}")
        return False


def setup_rotating_logger(
    name: str,
    log_file: str,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 5,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configura um logger com rotação automática de arquivos.

    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log
        max_bytes: Tamanho máximo do arquivo antes de rotacionar
        backup_count: Número de arquivos de backup a manter
        level: Nível de logging

    Returns:
        Logger configurado
    """
    # Criar diretório se necessário
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Obter ou criar logger
    log = logging.getLogger(name)
    log.setLevel(level)

    # Evitar handlers duplicados
    if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        # Handler de arquivo com rotação
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        # Formato detalhado
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log
