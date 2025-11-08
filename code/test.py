import logging
from utilsarma.core import DataProcessor
from pathlib import Path

# ========= 日志配置 =========
LOG_FILE = "arma.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent / LOG_FILE, encoding="utf-8"
        ),  # 写入文件
        # logging.StreamHandler(),  # 同时在控制台显示
    ],
)

# ===== 使用 DataProcessor =====
DP = DataProcessor(Path(__file__).parent.parent / "data" / "BTCUSDT_1d.parquet")
DP.load_data()
logging.info(f"{DP.table}")
