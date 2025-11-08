import logging
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

logger = logging.getLogger("ARMA_LOG")  # 注意这里匹配主程序的名字


class DataProcessor:
    def __init__(self, path: Path) -> None:
        self.table: pa.Table
        self.path: Path = path

    def load_data(self) -> None:
        self.table = pq.read_table(self.path)
        logger.info(
            f"KlineManager: 加载 {self.path} 数据，共 {self.table.num_rows} 条记录。"
        )
