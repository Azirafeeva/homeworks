from pathlib import Path
import yaml
from typing import Any


class Config:
    """Менеджер конфигурации проекта."""
    
    def __init__(self):
        # Корень проекта (на 1 уровень выше src/)
        self.project_root = Path(__file__).parent.parent
        
        # Загружаем YAML
        config_path = self.project_root / 'configs' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def data_raw_path(self) -> Path:
        """Абсолютный путь к сырым данным."""
        return self.project_root / self._config['paths']['data_raw']
    
    @property
    def data_processed_path(self) -> Path:
        """Абсолютный путь к обработанным данным."""
        return self.project_root / self._config['paths']['data_processed']
    
    @property
    def models_dir(self) -> Path:
        """Абсолютный путь к папке моделей."""
        return self.project_root / self._config['paths']['models']
    
    @property
    def figures_dir(self) -> Path:
        """Абсолютный путь к папке рисунков."""
        return self.project_root / self._config['paths']['figures']
    
    @property
    def reports_dir(self) -> Path:
        """Абсолютный путь к папке отчётов."""
        return self.project_root / self._config['paths']['reports']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение по ключу (поддержка вложенности через точку)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def ensure_dirs(self):
        """Создаёт все необходимые директории."""
        for dir_path in [self.models_dir, self.figures_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр конфига
config = Config()
config.ensure_dirs()