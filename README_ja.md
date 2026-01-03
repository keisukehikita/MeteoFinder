# MeteoFinder

OpenCVによる事前フィルタリングとClaude Vision APIを使用して、夜空の写真から流星を検出するハイブリッドツールです。

## 仕組み

1. **事前フィルタ (OpenCV)**: エッジ検出とハフ変換を使用して、画像内の直線的な軌跡を高速でスキャン
2. **検証 (Claude API)**: 候補画像をClaude Vision APIに送信し、流星の存在を確認。飛行機、人工衛星、星の軌跡などを除外

このハイブリッドアプローチにより、全画像をAPIに送信する場合と比較してAPIコストを約85%削減できます。

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/keisukehikita/MeteoFinder.git
cd MeteoFinder

# 依存パッケージをインストール
pip install -r requirements.txt

# APIキーを設定
cp config.example.py config.py
# config.pyを編集してAnthropic APIキーを追加
```

## 使い方

```bash
# 基本的な使い方（APIキー設定済みならAPI使用、なければローカルのみ）
python main.py /path/to/your/photos

# 事前フィルタの感度調整（1-5、デフォルト: 3）
python main.py /path/to/photos --sensitivity 4

# ローカルモードを強制（API呼び出しなし、高速だが精度は低下）
python main.py /path/to/photos --local

# 事前フィルタのみ（コピーせず候補をリスト表示）
python main.py /path/to/photos --prefilter-only
```

## モード

| モード | 説明 | 精度 |
|--------|------|------|
| ハイブリッド（APIキー設定時のデフォルト） | OpenCV事前フィルタ + Claude API検証 | 最高 |
| ローカルのみ（APIキーなし または --local） | OpenCV検出のみ | 良好、誤検出の可能性あり |
| 事前フィルタのみ | コピーせず候補をリスト表示 | テスト用 |

### 感度レベル

| レベル | 説明 | 推定API呼び出し |
|--------|------|-----------------|
| 1 | 非常に厳格 - 明らかな軌跡のみ | 画像の約5% |
| 2 | 厳格 | 約10% |
| 3 | バランス（推奨） | 約15% |
| 4 | 敏感 | 約25% |
| 5 | 非常に敏感 - 淡い軌跡も検出 | 約35%以上 |

## 出力

検出された流星画像は、スキャンしたフォルダ内の `Found/` サブフォルダにコピーされます。

```
YourPhotos/
├── IMG_001.jpg
├── IMG_002.jpg
├── ...
└── Found/           # 流星画像はここにコピー
    └── IMG_042.jpg
```

## コスト目安

感度3で1晩500枚の画像を処理する場合:
- 約75枚がAPIに送信（15%が事前フィルタを通過）
- コスト: 約$0.50/晩

## 必要条件

- Python 3.10以上
- Anthropic APIキー（[こちらで取得](https://console.anthropic.com/)）

## ライセンス

MIT
