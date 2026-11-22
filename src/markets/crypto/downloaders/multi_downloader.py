import sys
import os

# Добавляем корень проекта в путь
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from src.markets.crypto.downloaders.base_downloader import BaseDownloader


class MultiDownloader:
    def __init__(self):
        self.downloader = BaseDownloader()

    def update_all(self):
        assets = [
            {"ticker": "BTC-USD", "asset_id": 1, "name": "Bitcoin"},
            {"ticker": "ETH-USD", "asset_id": 2, "name": "Ethereum"},
            {"ticker": "SOL-USD", "asset_id": 3, "name": "Solana"}
        ]

        total = 0
        for asset in assets:
            count = self.downloader.download_and_save(
                ticker=asset["ticker"], 
                asset_id=asset["asset_id"]
            )
            total += count

        self.downloader.close()
        print(f"\n🎉 Обновление завершено. Всего загружено записей: {total}")

    def update_single(self, ticker: str, asset_id: int):
        self.downloader.download_and_save(ticker, asset_id)
        self.downloader.close()


if __name__ == "__main__":
    downloader = MultiDownloader()
    downloader.update_all()